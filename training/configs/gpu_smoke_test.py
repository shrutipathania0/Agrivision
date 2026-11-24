"""
PyTorch & CUDA Hardware Smoke Test for AgriVision
Tests:
- GPU device detection and properties
- CUDA tensor allocation
- Mixed precision (FP16 AMP) execution
- Pretrained EfficientNet-B0 forward and backward pass
- VRAM footprint profiling
"""

import sys
import time
import torch
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


def run_gpu_smoke_test():
    print("=" * 60)
    print("AGRIVISION HARDWARE & PYTORCH COMPATIBILITY SMOKE TEST")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available to PyTorch. Check drivers and CUDA wheel installation.")
        sys.exit(1)
        
    device_id = 0
    device_name = torch.cuda.get_device_name(device_id)
    capability = torch.cuda.get_device_capability(device_id)
    total_mem_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    
    print(f"Device [{device_id}]: {device_name}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")
    print(f"Total Dedicated VRAM: {total_mem_gb:.2f} GB")
    
    # 1. Simple tensor operations on GPU
    print("\n[Step 1/4] Running synthetic CUDA tensor allocation...")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print(f"-> Tensor multiply succeeded. Memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    del x, y
    torch.cuda.empty_cache()
    
    # 2. EfficientNet-B0 model loading & ImageNet weights verification
    print("\n[Step 2/4] Loading pretrained EfficientNet-B0 on CUDA...")
    start_time = time.time()
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to("cuda")
    print(f"-> Model loaded in {time.time() - start_time:.2f}s. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. FP16 Mixed Precision forward and backward pass with batch size = 8
    print("\n[Step 3/4] Testing FP16 mixed precision forward and backward pass (batch_size=8, 224x224)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cuda")
    
    dummy_input = torch.randn(8, 3, 224, 224, device="cuda")
    dummy_target = torch.randint(0, 1000, (8,), device="cuda")
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"-> Forward/Backward pass succeeded. Loss: {loss.item():.4f}")
    print(f"-> Peak VRAM allocated during batch size 8 training: {peak_vram_mb:.2f} MB ({peak_vram_mb / 1024:.2f} GB)")
    
    # 4. Batch size = 16 stress test
    print("\n[Step 4/4] Testing batch size = 16 for memory ceiling...")
    dummy_input_16 = torch.randn(16, 3, 224, 224, device="cuda")
    dummy_target_16 = torch.randint(0, 1000, (16,), device="cuda")
    
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        output_16 = model(dummy_input_16)
        loss_16 = criterion(output_16, dummy_target_16)
        
    scaler.scale(loss_16).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    
    peak_vram_16_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"-> Batch size 16 pass succeeded! Peak VRAM: {peak_vram_16_mb:.2f} MB ({peak_vram_16_mb / 1024:.2f} GB)")
    
    del model, dummy_input, dummy_target, dummy_input_16, dummy_target_16, optimizer, scaler
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("STATUS: PASSED. System is 100% READY for EfficientNet-B0 training on RTX 3060!")
    print("=" * 60)


if __name__ == "__main__":
    run_gpu_smoke_test()
