'use client';

import React from 'react';
import {
  Sparkles,
  GitCompare,
  Cpu,
  CheckCircle2,
  ShieldAlert,
  ArrowRight,
  Database,
  Layers,
} from 'lucide-react';

interface EmptyWorkspaceProps {
  onSelectQuery: (query: string) => void;
  isLoading: boolean;
}

export const CURATED_RESEARCH_QUESTIONS = [
  {
    category: 'Comparative Architecture',
    query: 'How do RAG-Sequence and RAG-Token differ in generating sequences?',
    paperTag: 'P1: RAG',
    badgeColor: 'text-indigo-400 bg-indigo-950/50 border-indigo-800/40',
    description:
      'Analyzes latent variable sequence-level vs. token-level marginalization over retrieved documents.',
    icon: GitCompare,
  },
  {
    category: 'Adaptive Reflection Tokens',
    query: 'What are the four reflection tokens in Self-RAG and what does each one assess?',
    paperTag: 'P4: Self-RAG',
    badgeColor: 'text-violet-400 bg-violet-950/50 border-violet-800/40',
    description:
      'Inspects inline critique tokens (Retrieve, ISREL, ISSUP, ISUSE) governing selective retrieval.',
    icon: Cpu,
  },
  {
    category: 'Dual-Encoder Negative Sampling',
    query: 'How does DPR train its passage encoder, and what types of negatives are used?',
    paperTag: 'P2: DPR',
    badgeColor: 'text-emerald-400 bg-emerald-950/50 border-emerald-800/40',
    description:
      'Examines in-batch negatives and BM25 hard negatives used to train dense passage representations.',
    icon: Layers,
  },
  {
    category: 'Automated Evaluation Metric',
    query: 'How does RAGAS compute its faithfulness metric?',
    paperTag: 'P5: Ragas',
    badgeColor: 'text-blue-400 bg-blue-950/50 border-blue-800/40',
    description:
      'Deconstructs answer statements into atomic claims to calculate context support ratios.',
    icon: CheckCircle2,
  },
  {
    category: 'Safety & Abstention Safeguard',
    query: "Who won the 2024 ICC Men's T20 World Cup?",
    paperTag: 'Out-of-Scope',
    badgeColor: 'text-slate-400 bg-slate-900 border-slate-700/50',
    description:
      'Validates that out-of-scope queries return explicit abstention with strictly zero hallucinated citations.',
    icon: ShieldAlert,
  },
];

export const EmptyWorkspace: React.FC<EmptyWorkspaceProps> = ({ onSelectQuery, isLoading }) => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto py-4 animate-fade-in">
      {/* Editorial Hero Intro */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 text-xs font-medium">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Grounded Scientific Synthesis • 396 Indexed Chunks</span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold text-white tracking-tight">
          Explore Scientific Literature with Verified Citations
        </h2>
        <p className="text-xs md:text-sm text-[#94a3b8] max-w-xl mx-auto leading-relaxed">
          Every statement is synthesized from peer-reviewed evidence indexed in Qdrant, verified against source passages, and bound to exact physical page numbers.
        </p>
      </div>

      {/* Curated Questions Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8] flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-indigo-400" />
            Curated Benchmark Inquiries
          </span>
          <span className="text-[11px] text-[#64748b]">Select a question to populate and explore</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {CURATED_RESEARCH_QUESTIONS.map((item, idx) => {
            const Icon = item.icon;
            return (
              <button
                key={idx}
                type="button"
                disabled={isLoading}
                onClick={() => onSelectQuery(item.query)}
                className="p-4 rounded-xl bg-[#111318] hover:bg-[#171a21] border border-[#232732] hover:border-indigo-500/40 text-left transition duration-150 flex flex-col justify-between gap-3 group cursor-pointer disabled:opacity-50 shadow-sm"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-[11px] font-medium text-[#64748b] flex items-center gap-1.5">
                      <Icon className="w-3.5 h-3.5 text-indigo-400" />
                      {item.category}
                    </span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded border font-semibold ${item.badgeColor}`}
                    >
                      {item.paperTag}
                    </span>
                  </div>

                  <h3 className="text-xs font-semibold text-white group-hover:text-indigo-300 transition line-clamp-2 leading-snug">
                    {item.query}
                  </h3>

                  <p className="text-[11px] text-[#94a3b8] mt-1.5 leading-relaxed line-clamp-2">
                    {item.description}
                  </p>
                </div>

                <div className="flex items-center justify-end text-[11px] text-indigo-400 font-medium gap-1 pt-2 border-t border-[#232732]/60 group-hover:text-indigo-300">
                  <span>Load question</span>
                  <ArrowRight className="w-3.5 h-3.5 transition group-hover:translate-x-0.5" />
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
