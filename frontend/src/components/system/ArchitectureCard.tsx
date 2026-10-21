'use client';

import React from 'react';
import { Layers, Database, Search, ShieldCheck, FileText } from 'lucide-react';

export const ArchitectureCard: React.FC = () => {
  const stages = [
    {
      num: '1',
      title: 'Ingestion & Pagination',
      tech: 'PyMuPDF',
      desc: 'Page-aware PDF extraction preserving physical page numbers and sentence boundaries.',
      icon: FileText,
      color: 'text-indigo-400',
      bg: 'bg-indigo-950/30 border-indigo-800/40',
    },
    {
      num: '2',
      title: 'Dual Indexing',
      tech: 'Qdrant + BM25',
      desc: '384d SBERT embeddings in embedded Qdrant plus tokenized Okapi BM25 inverted index.',
      icon: Database,
      color: 'text-emerald-400',
      bg: 'bg-emerald-950/30 border-emerald-800/40',
    },
    {
      num: '3',
      title: 'Hybrid Fusion',
      tech: 'RRF (k=60) + Router',
      desc: 'Reciprocal Rank Fusion combining dense semantic and sparse lexical ranks with opt-in intent routing.',
      icon: Search,
      color: 'text-violet-400',
      bg: 'bg-violet-950/30 border-violet-800/40',
    },
    {
      num: '4',
      title: 'Evidence Synthesis',
      tech: 'Multi-LLM + Citations',
      desc: 'Page-level citation provenance verification with guaranteed abstention safeguards.',
      icon: ShieldCheck,
      color: 'text-amber-400',
      bg: 'bg-amber-950/30 border-amber-800/40',
    },
  ];

  return (
    <div className="border border-[#232732] rounded-xl bg-[#111318] overflow-hidden shadow-lg p-5 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[#232732]">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-violet-400" />
            Engine Architectural Pipeline
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Strict evidence grounding with page-level citations from scientific papers
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {stages.map((stage) => {
          const Icon = stage.icon;
          return (
            <div
              key={stage.num}
              className={`p-3.5 rounded-lg border flex flex-col justify-between gap-3 ${stage.bg}`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="w-6 h-6 rounded-full bg-[#111318] border border-[#232732] text-[11px] font-mono font-bold text-white flex items-center justify-center">
                    {stage.num}
                  </span>
                  <Icon className={`w-4 h-4 ${stage.color}`} />
                </div>
                <h4 className="text-xs font-semibold text-white">{stage.title}</h4>
                <span className="text-[10px] font-mono font-semibold text-indigo-300 block mb-1">
                  {stage.tech}
                </span>
                <p className="text-[11px] text-[#94a3b8] leading-relaxed">
                  {stage.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
