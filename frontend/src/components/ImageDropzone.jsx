import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, Camera, AlertCircle, RefreshCw, CheckCircle2 } from 'lucide-react';

export default function ImageDropzone({ selectedImage, onImageSelected, onClearImage, onAnalyze, isAnalyzing }) {
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [resolutionWarning, setResolutionWarning] = useState(null);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPEG, PNG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        if (img.width < 224 || img.height < 224) {
          setResolutionWarning(`Image resolution (${img.width}x${img.height}) is relatively low. For highest accuracy, capture at least 224x224 pixels.`);
        } else {
          setResolutionWarning(null);
        }
        onImageSelected(file, e.target.result, { width: img.width, height: img.height });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-5 sm:p-6 transition">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-emerald-600" />
            Upload Crop Leaf Image
          </h2>
          <p className="text-xs sm:text-sm text-slate-500">
            Capture or upload a clear, focused photo of a single diseased or healthy leaf
          </p>
        </div>
        {selectedImage && (
          <button
            onClick={onClearImage}
            disabled={isAnalyzing}
            className="text-xs text-rose-600 hover:text-rose-700 font-medium px-2.5 py-1 rounded-md hover:bg-rose-50 border border-rose-200 transition"
          >
            Clear Image
          </button>
        )}
      </div>

      {!selectedImage ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`border-2 border-dashed rounded-xl p-8 sm:p-10 text-center transition cursor-pointer flex flex-col items-center justify-center ${
            isDragOver
              ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]'
              : 'border-slate-300 hover:border-emerald-500 hover:bg-slate-50/60'
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mb-4 text-emerald-700 shadow-sm">
            <UploadCloud className="w-8 h-8" />
          </div>

          <h3 className="text-base font-semibold text-slate-800 mb-1">
            Drag & drop leaf photo here, or browse
          </h3>
          <p className="text-xs text-slate-500 max-w-sm mb-5">
            Supports Potato, Tomato, Rice, and Wheat leaves. Formats: JPG, PNG, WEBP (up to 15MB).
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white text-xs sm:text-sm font-semibold rounded-lg shadow-sm transition flex items-center gap-2"
            >
              <UploadCloud className="w-4 h-4" />
              Select File
            </button>

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                cameraInputRef.current?.click();
              }}
              className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs sm:text-sm font-semibold rounded-lg border border-slate-200 shadow-sm transition flex items-center gap-2"
            >
              <Camera className="w-4 h-4 text-slate-600" />
              Use Camera
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
          />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden bg-slate-900 border border-slate-200 aspect-video sm:aspect-[21/9] flex items-center justify-center">
            <img
              src={selectedImage.previewUrl}
              alt="Leaf Preview"
              className="max-h-full max-w-full object-contain mx-auto"
            />
            <div className="absolute top-3 left-3 bg-slate-900/80 backdrop-blur-sm text-white text-xs px-2.5 py-1 rounded-md font-mono flex items-center gap-1.5 border border-white/10">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              {selectedImage.file.name} ({selectedImage.dimensions.width}×{selectedImage.dimensions.height}px)
            </div>
          </div>

          {resolutionWarning && (
            <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 text-amber-800 p-3 rounded-lg text-xs">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <span>{resolutionWarning}</span>
            </div>
          )}

          <button
            onClick={onAnalyze}
            disabled={isAnalyzing}
            className={`w-full py-3.5 px-6 rounded-xl font-bold text-sm sm:text-base text-white shadow-md transition flex items-center justify-center gap-2.5 ${
              isAnalyzing
                ? 'bg-emerald-400 cursor-not-allowed'
                : 'bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99] shadow-emerald-700/20'
            }`}
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                <span>Running Deep Learning Inference & Grad-CAM Analysis...</span>
              </>
            ) : (
              <>
                <ImageIcon className="w-5 h-5" />
                <span>Run AI Disease Diagnosis</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
