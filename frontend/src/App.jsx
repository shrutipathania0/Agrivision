import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ImageDropzone from './components/ImageDropzone';
import DiagnosticResult from './components/DiagnosticResult';
import AgriChatbot from './components/AgriChatbot';
import SupportedCropsModal from './components/SupportedCropsModal';
import ModelEvaluation from './components/ModelEvaluation';
import { AlertCircle, CheckCircle2, Shield, Leaf, Activity, BarChart3, ScanLine } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('diagnose'); // 'diagnose' | 'evaluation'
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [currentScanContext, setCurrentScanContext] = useState(null);
  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [isCropsModalOpen, setIsCropsModalOpen] = useState(false);
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', device: 'unknown' });

  // Check backend health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get('/api/health');
        setBackendHealth(res.data);
      } catch (err) {
        setBackendHealth({ status: 'offline', device: 'none' });
      }
    };
    checkHealth();
  }, []);

  const handleImageSelected = (file, previewUrl, dimensions) => {
    setSelectedImage({ file, previewUrl, dimensions });
    setAnalysisResult(null); // Clear previous scan
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setAnalysisResult(null);
    setCurrentScanContext(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage || isAnalyzing) return;

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedImage.file);

      const response = await axios.post('/api/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const data = response.data;
      setAnalysisResult(data);

      // Seed context for the agricultural chatbot
      setCurrentScanContext({
        crop: data.crop,
        disease: data.disease,
        confidence: data.confidence
      });
    } catch (err) {
      console.error('Detection error:', err);
      const errMsg = err.response?.data?.detail || 'Failed to analyze leaf image. Please check backend connection.';
      alert(`Diagnosis Error: ${errMsg}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  const handleDownloadReport = async () => {
    if (!analysisResult || isDownloadingReport) return;

    setIsDownloadingReport(true);
    try {
      let origBase64 = null;
      if (selectedImage?.file) {
        try {
          origBase64 = await fileToBase64(selectedImage.file);
        } catch (e) {
          console.warn('Could not encode original image to base64:', e);
        }
      }

      const payload = {
        crop: analysisResult.crop,
        disease: analysisResult.disease,
        confidence: analysisResult.confidence,
        is_healthy: analysisResult.is_healthy,
        is_ambiguous: analysisResult.is_ambiguous,
        entropy: analysisResult.entropy,
        pathogen: analysisResult.pathogen,
        advisory: analysisResult.advisory,
        gradcam_base64: analysisResult.gradcam_base64,
        original_image_base64: origBase64,
        top_alternatives: analysisResult.top_alternatives,
        client_timestamp: new Date().toISOString()
      };

      const response = await axios.post('/api/report', payload, {
        responseType: 'blob'
      });

      // Create download trigger
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `AgriVision_Detailed_Report_${analysisResult.crop}_${analysisResult.disease.replace(/\s+/g, '_')}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF generation error:', err);
      alert('Failed to generate diagnostic PDF report.');
    } finally {
      setIsDownloadingReport(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900">
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onOpenCropsModal={() => setIsCropsModalOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {/* Top Status & Overview Banner */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <div className="flex items-center gap-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs sm:text-sm font-semibold text-slate-800">
              AgriVision Dual-AI Engine Active
            </span>
            <span className="text-xs text-slate-400">|</span>
            <span className="text-xs text-slate-500 font-mono">
              Inference Hardware: {backendHealth.device ? backendHealth.device.toUpperCase() : 'RTX 3060'}
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-600">
            <span className="flex items-center gap-1 font-medium">
              <Leaf className="w-3.5 h-3.5 text-emerald-600" />
              4 Staple Crops
            </span>
            <span className="text-slate-300">•</span>
            <span className="flex items-center gap-1 font-medium">
              <Shield className="w-3.5 h-3.5 text-emerald-600" />
              PAU & ICAR Grounded
            </span>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'diagnose' ? (
          <>
            {/* Upload & Diagnosis Section */}
            <section className="space-y-6">
              <ImageDropzone
                selectedImage={selectedImage}
                onImageSelected={handleImageSelected}
                onClearImage={handleClearImage}
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing}
              />

              {analysisResult && (
                <DiagnosticResult
                  result={analysisResult}
                  originalImageSrc={selectedImage?.previewUrl}
                  onDownloadReport={handleDownloadReport}
                  isDownloadingReport={isDownloadingReport}
                />
              )}
            </section>

            {/* Agricultural Chatbot Section */}
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-emerald-600" />
                  Interactive Farmer Advisory Chat
                </h2>
                <span className="text-xs text-slate-500 hidden sm:inline">
                  Ask follow-up questions about the diagnosed disease or farming guidelines
                </span>
              </div>

              <AgriChatbot
                currentScanContext={currentScanContext}
                onClearScanContext={() => setCurrentScanContext(null)}
              />
            </section>
          </>
        ) : (
          /* Model Evaluation Tab */
          <ModelEvaluation />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-12 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 space-y-2">
          <p className="font-medium text-slate-700">
            AgriVision: Smart Crop Disease Detection, Agricultural Advisory & AI Assistant
          </p>
          <p className="max-w-3xl mx-auto text-slate-400">
            Grounding references: Punjab Agricultural University (PAU) Ludhiana • CCS Haryana Agricultural University (CCS HAU) Hisar • ICAR-IIWBR Karnal • ICAR-CPRI Shimla • KVK Extension Network.
          </p>
          <p className="text-[11px] text-slate-400 pt-1">
            Disclaimer: AgriVision is an AI-assisted diagnostic advisory aid. Always confirm chemical controls with your local agricultural officer.
          </p>
        </div>
      </footer>

      {/* Supported Crops Modal */}
      <SupportedCropsModal
        isOpen={isCropsModalOpen}
        onClose={() => setIsCropsModalOpen(false)}
      />
    </div>
  );
}
