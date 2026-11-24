import React from 'react';
import { X, CheckCircle, MapPin, BookOpen, AlertCircle } from 'lucide-react';

export default function SupportedCropsModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const taxonomy = [
    {
      crop: 'Potato',
      scientific: 'Solanum tuberosum',
      season: 'Rabi (Oct - Feb)',
      regions: 'Jalandhar, Hoshiarpur, Kurukshetra, Yamunanagar',
      diseases: [
        { name: 'Early Blight', pathogen: 'Alternaria solani', status: 'Prevalent in early crop stage' },
        { name: 'Late Blight', pathogen: 'Phytophthora infestans', status: 'Critical threat during winter fog/dew' },
        { name: 'Healthy', pathogen: 'None', status: 'Baseline vigor' }
      ]
    },
    {
      crop: 'Tomato',
      scientific: 'Solanum lycopersicum',
      season: 'Autumn & Spring crops',
      regions: 'Amritsar, Patiala, Karnal, Sonipat',
      diseases: [
        { name: 'Early Blight', pathogen: 'Alternaria solani', status: 'Target spot concentric rings' },
        { name: 'Late Blight', pathogen: 'Phytophthora infestans', status: 'Water-soaked rapid foliar blight' },
        { name: 'Septoria Leaf Spot', pathogen: 'Septoria lycopersici', status: 'Gray centers with pycnidia' },
        { name: 'Leaf Mold', pathogen: 'Passalora fulva', status: 'Velvety olive growth in polyhouses' },
        { name: 'Bacterial Spot', pathogen: 'Xanthomonas spp.', status: 'Angular greasy water-soaked spots' },
        { name: 'Mosaic Virus', pathogen: 'ToMV', status: 'Mottled light/dark green distortion' },
        { name: 'Healthy', pathogen: 'None', status: 'Baseline vigor' }
      ]
    },
    {
      crop: 'Rice',
      scientific: 'Oryza sativa',
      season: 'Kharif (June - Nov)',
      regions: 'Basmati & Non-Basmati tracts across Punjab & Haryana',
      diseases: [
        { name: 'Bacterial Leaf Blight', pathogen: 'Xanthomonas oryzae pv. oryzae', status: 'Wavy translucent bleached stripes' },
        { name: 'Blast', pathogen: 'Magnaporthe oryzae', status: 'Spindle-shaped lesions & neck blast' },
        { name: 'Brown Spot', pathogen: 'Bipolaris oryzae', status: 'Oval spots with yellow halos' },
        { name: 'Leaf Smut', pathogen: 'Entyloma oryzae', status: 'Angular black crust spots' },
        { name: 'Healthy', pathogen: 'None', status: 'Baseline vigor' }
      ]
    },
    {
      crop: 'Wheat',
      scientific: 'Triticum aestivum',
      season: 'Rabi (Nov - April)',
      regions: 'Premier staple cereal across all districts',
      diseases: [
        { name: 'Yellow Rust (Stripe Rust)', pathogen: 'Puccinia striiformis', status: 'Quarantine priority in foothill districts' },
        { name: 'Leaf Rust (Brown Rust)', pathogen: 'Puccinia triticina', status: 'Scattered orange pustules in spring' },
        { name: 'Powdery Mildew', pathogen: 'Blumeria graminis', status: 'Fluffy white mold in dense canopy' },
        { name: 'Healthy', pathogen: 'None', status: 'Baseline vigor' }
      ]
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[85vh] shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-emerald-800 text-white">
          <div className="flex items-center gap-2.5">
            <BookOpen className="w-5 h-5 text-emerald-200" />
            <div>
              <h2 className="text-base font-bold">Supported Crops & Disease Taxonomy</h2>
              <p className="text-xs text-emerald-200/90">Validated against institutional field registries</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-emerald-100 hover:text-white rounded-lg hover:bg-emerald-700/60 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3.5 text-xs text-emerald-900 flex items-start gap-2.5">
            <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <p>
              AgriVision models are genuinely trained strictly on verified visual symptoms of these classes. Uploading leaves from unsupported species or unlisted diseases will trigger our uncertainty rejection filter.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {taxonomy.map((c, i) => (
              <div key={i} className="border border-slate-200 rounded-xl p-4 bg-slate-50/50 hover:bg-white hover:border-emerald-300 transition shadow-2xs">
                <div className="flex items-center justify-between mb-1.5">
                  <h3 className="text-base font-extrabold text-slate-900">{c.crop}</h3>
                  <span className="text-[11px] font-semibold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full">
                    {c.season}
                  </span>
                </div>
                <p className="text-xs text-slate-500 italic mb-2">{c.scientific}</p>

                <div className="flex items-center gap-1.5 text-[11px] text-slate-600 mb-3 bg-white p-2 rounded-lg border border-slate-100">
                  <MapPin className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  <span className="truncate">{c.regions}</span>
                </div>

                <div className="space-y-1.5">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                    Recognized Conditions:
                  </span>
                  {c.diseases.map((d, j) => (
                    <div key={j} className="text-xs flex items-center justify-between bg-white px-2.5 py-1.5 rounded-md border border-slate-200/70">
                      <span className="font-medium text-slate-800">{d.name}</span>
                      <span className="text-[10px] text-slate-400 truncate max-w-[140px]">{d.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-100 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
          <span>Sources: PAU Ludhiana, CCS HAU Hisar, ICAR-IIWBR, ICAR-CPRI</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-900 text-white font-medium rounded-lg transition"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
}
