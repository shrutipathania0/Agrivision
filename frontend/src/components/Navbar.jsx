import React from 'react';
import { Sprout, BookOpen, ShieldCheck, BarChart3, ScanLine } from 'lucide-react';

export default function Navbar({ activeTab, onSelectTab, onOpenCropsModal }) {
  return (
    <header className="bg-emerald-800 text-white shadow-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onSelectTab('diagnose')}>
            <div className="bg-emerald-600 p-2 rounded-xl shadow-inner flex items-center justify-center">
              <Sprout className="w-6 h-6 text-emerald-100" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight">AgriVision</span>
                <span className="text-xs bg-emerald-700/80 text-emerald-200 px-2 py-0.5 rounded-full font-medium border border-emerald-600">
                  v1.0 AI
                </span>
              </div>
              <p className="text-xs text-emerald-200/90 hidden sm:block">
                Smart Crop Disease Detection & Agricultural Advisory (Punjab & Haryana)
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center bg-emerald-900/70 p-1 rounded-xl border border-emerald-700/60">
            <button
              onClick={() => onSelectTab('diagnose')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold transition ${
                activeTab === 'diagnose'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-emerald-200 hover:text-white hover:bg-emerald-800/50'
              }`}
            >
              <ScanLine className="w-4 h-4" />
              <span>Crop Diagnosis</span>
            </button>

            <button
              onClick={() => onSelectTab('evaluation')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-semibold transition ${
                activeTab === 'evaluation'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-emerald-200 hover:text-white hover:bg-emerald-800/50'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Model Evaluation</span>
            </button>
          </nav>

          {/* Action Links & Badges */}
          <div className="flex items-center space-x-3">
            <div className="hidden lg:flex items-center space-x-1.5 text-xs bg-emerald-900/60 text-emerald-200 px-3 py-1.5 rounded-lg border border-emerald-700/50">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>PAU & ICAR Grounded</span>
            </div>

            <button
              onClick={onOpenCropsModal}
              className="flex items-center space-x-1.5 text-xs sm:text-sm bg-emerald-700 hover:bg-emerald-600 text-white px-3.5 py-1.5 rounded-lg font-medium transition shadow-sm border border-emerald-600 active:scale-95"
            >
              <BookOpen className="w-4 h-4 text-emerald-200" />
              <span className="hidden sm:inline">Supported Crops</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
