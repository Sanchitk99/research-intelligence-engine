'use client';

import React from 'react';
import { Sliders, Database, Layers } from 'lucide-react';
import { EngineSettings } from '@/lib/types';

interface HeaderProps {
  currentTab: 'research' | 'papers' | 'status' | 'about';
  settings: EngineSettings;
  onOpenSettings: () => void;
  totalPoints?: number;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  settings,
  onOpenSettings,
  totalPoints = 396,
}) => {
  const getTabMeta = () => {
    switch (currentTab) {
      case 'research':
        return {
          title: 'Evidence-Grounded Research Workspace',
          subtitle: 'Synthesizing verified claims with page-level citations from the corpus',
        };
      case 'papers':
        return {
          title: 'Paper Library & Ingestion',
          subtitle: 'Corpus index management, boundary-aware chunking, and PDF ingestion',
        };
      case 'status':
        return {
          title: 'Engine Architecture & System Status',
          subtitle: 'Live vector storage metrics, provider readiness, and retrieval health',
        };
      case 'about':
        return {
          title: 'About This Project',
          subtitle: 'Architecture flow, empirical retrieval evaluation, and technology stack',
        };
    }
  };

  const meta = getTabMeta();

  return (
    <header className="h-16 border-b border-[#232732] bg-[#111318]/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
      <div>
        <h2 className="text-sm font-semibold text-white tracking-tight">{meta.title}</h2>
        <p className="text-xs text-[#94a3b8]">{meta.subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Active Retrieval Badge */}
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#171a21] border border-[#232732] text-[11px] text-[#94a3b8]">
          <Layers className="w-3 h-3 text-indigo-400" />
          <span>Mode:</span>
          <strong className="text-white uppercase font-mono">{settings.retrieval_mode}</strong>
          <span className="text-[#64748b]">•</span>
          <span>k={settings.top_k}</span>
        </div>

        {/* Corpus Points Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#171a21] border border-[#232732] text-[11px] text-[#94a3b8]">
          <Database className="w-3 h-3 text-emerald-400" />
          <span>Corpus:</span>
          <strong className="text-white font-mono">{totalPoints}</strong>
          <span>chunks</span>
        </div>

        {/* Settings button */}
        <button
          onClick={onOpenSettings}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#171a21] hover:bg-[#232732] text-white border border-[#232732] hover:border-[#3b4455] text-xs font-medium transition cursor-pointer"
        >
          <Sliders className="w-3.5 h-3.5 text-indigo-400" />
          <span>Config</span>
        </button>
      </div>
    </header>
  );
};
