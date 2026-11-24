import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Eye,
  Sliders,
  FileDown,
  AlertTriangle,
  Info,
  CheckCircle,
  Stethoscope,
  ChevronDown,
  ChevronUp,
  ExternalLink
} from 'lucide-react';

export default function DiagnosticResult({
  result,
  originalImageSrc,
  onDownloadReport,
  isDownloadingReport
}) {
  const [showGradCam, setShowGradCam] = useState(true);
  const [gradCamOpacity, setGradCamOpacity] = useState(0.65);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [expandedSection, setExpandedSection] = useState('symptoms');

  if (!result) return null;

  const isHealthy = result.is_healthy;
  const isAmbiguous = result.is_ambiguous;
  const confidencePercent = Math.round(result.confidence * 100);

  // Status color helpers
  const getConfidenceColor = (conf) => {
    if (conf >= 85) return 'text-emerald-700 bg-emerald-50 border-emerald-200';
    if (conf >= 65) return 'text-amber-700 bg-amber-50 border-amber-200';
    return 'text-rose-700 bg-rose-50 border-rose-200';
  };

  const getStatusBadge = () => {
    if (isAmbiguous) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-800 border border-amber-300">
          <AlertTriangle className="w-3.5 h-3.5" />
          Low Confidence / Needs Review
        </span>
      );
    }
    if (isHealthy) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
          <ShieldCheck className="w-3.5 h-3.5" />
          Healthy Foliage
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-100 text-rose-800 border border-rose-300">
        <ShieldAlert className="w-3.5 h-3.5" />
        Disease Detected
      </span>
    );
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Top Banner: Diagnosis Result */}
      <div className="p-5 sm:p-6 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-xs font-bold tracking-wider uppercase text-slate-400">
                Crop Diagnosis
              </span>
              {getStatusBadge()}
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              {result.crop} — <span className="text-emerald-700">{result.disease}</span>
            </h2>
            {result.pathogen && (
              <p className="text-xs text-slate-500 italic mt-0.5">
                Causal Pathogen: {result.pathogen}
              </p>
            )}
          </div>

          {/* Confidence Score Pill */}
          <div className="flex items-center gap-3">
            <div className={`px-4 py-2 rounded-xl border text-center ${getConfidenceColor(confidencePercent)}`}>
              <div className="text-xs font-medium uppercase tracking-wider">AI Confidence</div>
              <div className="text-2xl font-black">{confidencePercent}%</div>
            </div>

            <button
              onClick={onDownloadReport}
              disabled={isDownloadingReport}
              className="px-4 py-2.5 bg-emerald-700 hover:bg-emerald-800 text-white rounded-xl font-bold text-xs sm:text-sm shadow-sm transition flex items-center gap-2 active:scale-95 disabled:bg-slate-300 disabled:cursor-not-allowed"
              title="Download Detailed Agricultural Pathology PDF Report"
            >
              <FileDown className="w-4 h-4 text-emerald-200" />
              <span>{isDownloadingReport ? 'Generating PDF...' : 'Download Detailed Report (PDF)'}</span>
            </button>
          </div>
        </div>

        {/* Ambiguous Warning Alert */}
        {isAmbiguous && (
          <div className="mt-4 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 flex items-start gap-3 text-xs sm:text-sm">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-bold">Uncertainty Guardrail Warning</p>
              <p className="mt-0.5 text-amber-800">
                The prediction confidence ({confidencePercent}%) is below our rigorous verification threshold (65%). This may be caused by background clutter, shadows, or an unsupported plant. Please capture a clear, well-lit close-up of a single leaf.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Main Content Grid: Image Visualizer (Left) & Advisory (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-0 divide-y lg:divide-y-0 lg:divide-x divide-slate-100">
        {/* Left Column: Grad-CAM Explainability Viewer */}
        <div className="lg:col-span-5 p-5 sm:p-6 bg-slate-50/50 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
                <Eye className="w-4 h-4 text-emerald-600" />
                Explainability (Grad-CAM Saliency)
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowGradCam(!showGradCam)}
                  className={`text-xs px-2.5 py-1 rounded-md font-semibold transition border ${
                    showGradCam
                      ? 'bg-emerald-600 text-white border-emerald-600'
                      : 'bg-white text-slate-600 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  {showGradCam ? 'Heatmap ON' : 'Original Only'}
                </button>
              </div>
            </div>

            {/* Visualizer Frame with Blend Overlay */}
            <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-square shadow-inner flex items-center justify-center">
              {/* Base Image */}
              <img
                src={originalImageSrc}
                alt="Original Leaf"
                className="w-full h-full object-contain"
              />

              {/* Grad-CAM Heatmap Layer */}
              {showGradCam && result.gradcam_base64 && (
                <img
                  src={`data:image/jpeg;base64,${result.gradcam_base64}`}
                  alt="Grad-CAM Activation"
                  style={{ opacity: gradCamOpacity }}
                  className="absolute inset-0 w-full h-full object-contain mix-blend-screen pointer-events-none transition-opacity duration-150"
                />
              )}
            </div>

            {/* Transparency Slider */}
            {showGradCam && result.gradcam_base64 && (
              <div className="mt-3 flex items-center gap-3 bg-white p-2.5 rounded-lg border border-slate-200">
                <Sliders className="w-4 h-4 text-slate-400" />
                <span className="text-xs text-slate-600 font-medium whitespace-nowrap">Heatmap Blend:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={gradCamOpacity}
                  onChange={(e) => setGradCamOpacity(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
                <span className="text-xs font-mono font-bold text-slate-600">
                  {Math.round(gradCamOpacity * 100)}%
                </span>
              </div>
            )}

            <p className="text-[11px] text-slate-500 mt-3 leading-relaxed">
              <span className="font-semibold text-slate-700">Scientific Interpretation:</span> Grad-CAM visualizes gradients flowing into the final convolutional layer of EfficientNet-B0. Warm highlighted zones indicate areas influencing the neural network’s disease classification.
            </p>
          </div>

          {/* Alternative Predictions Drawer */}
          {result.top_alternatives && result.top_alternatives.length > 1 && (
            <div className="mt-5 pt-4 border-t border-slate-200">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-2">
                Top Probabilities
              </span>
              <div className="space-y-1.5">
                {result.top_alternatives.slice(0, 3).map((alt, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-slate-700 truncate max-w-[70%]">{alt.class_name}</span>
                    <span className="font-mono font-semibold text-slate-500">
                      {Math.round(alt.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Agronomic Advisory Engine */}
        <div className="lg:col-span-7 p-5 sm:p-6 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Stethoscope className="w-5 h-5 text-emerald-600" />
                Integrated Crop Advisory
              </h3>
              <span className="text-xs text-emerald-800 bg-emerald-50 px-2.5 py-1 rounded-md font-medium border border-emerald-200">
                Punjab & Haryana Focus
              </span>
            </div>

            {/* Disease Explanation Card */}
            {result.advisory?.symptoms && (
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/80">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-1 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5 text-emerald-600" />
                  Diagnostic Symptoms
                </h4>
                <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
                  {result.advisory.symptoms}
                </p>
              </div>
            )}

            {/* Preventative Practices */}
            {result.advisory?.prevention_practices && (
              <div className="bg-emerald-50/50 rounded-xl p-4 border border-emerald-100">
                <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-800 mb-2 flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                  Preventive Agronomic Measures
                </h4>
                <ul className="space-y-1.5">
                  {result.advisory.prevention_practices.map((item, idx) => (
                    <li key={idx} className="text-xs sm:text-sm text-slate-700 flex items-start gap-2">
                      <span className="text-emerald-600 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Field Management & Sanitation */}
            {result.advisory?.management_practices && (
              <div className="bg-white rounded-xl p-4 border border-slate-200">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-2 flex items-center gap-1.5">
                  <Stethoscope className="w-3.5 h-3.5 text-emerald-600" />
                  Cultural Sanitation & Field Control
                </h4>
                <ul className="space-y-1.5">
                  {result.advisory.management_practices.map((item, idx) => (
                    <li key={idx} className="text-xs sm:text-sm text-slate-700 flex items-start gap-2">
                      <span className="text-emerald-700 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Safety & Institutional Grounding Disclaimer */}
            {result.advisory?.safety_warning && (
              <div className="bg-rose-50/70 rounded-xl p-3.5 border border-rose-200 text-rose-900 text-xs">
                <span className="font-bold block mb-0.5">⚠️ Mandatory Chemical Stewardship Notice:</span>
                <p className="text-rose-800 leading-relaxed">
                  {result.advisory.safety_warning} Adhere strictly to CIB&RC approved labels. Verify local symptoms with your block Krishi Vigyan Kendra (KVK) officer before spraying.
                </p>
              </div>
            )}
          </div>

          {/* Sources Attribution */}
          {result.advisory?.sources && (
            <div className="mt-5 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
              <span className="font-semibold text-slate-700">Verified Sources:</span>
              {result.advisory.sources.map((src, i) => (
                <span key={i} className="bg-slate-100 px-2 py-0.5 rounded text-slate-600 border border-slate-200">
                  {src.institution}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
