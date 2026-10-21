'use client';

import React from 'react';
import { Database, BookOpen, Layers } from 'lucide-react';
import { SystemStatus } from '@/lib/types';

interface SystemOverviewProps {
  status: SystemStatus | null;
  isLoading: boolean;
}

export const SystemOverview: React.FC<SystemOverviewProps> = ({ status, isLoading }) => {
  if (isLoading || !status) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-[#111318] rounded-xl border border-[#232732] animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {/* Total Chunks Card */}
      <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] shadow-lg flex items-center justify-between">
        <div>
          <span className="text-[10px] font-mono uppercase text-[#64748b] tracking-wider block mb-1">
            Qdrant Vector Points
          </span>
          <div className="text-2xl font-bold text-white font-mono">
            {status.total_points}
          </div>
          <span className="text-[11px] text-emerald-400 font-medium mt-0.5 block">
            Collection: {status.collection_name}
          </span>
        </div>
        <div className="w-10 h-10 rounded-lg bg-emerald-950/40 border border-emerald-700/50 flex items-center justify-center text-emerald-400">
          <Database className="w-5 h-5" />
        </div>
      </div>

      {/* Indexed Papers Card */}
      <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] shadow-lg flex items-center justify-between">
        <div>
          <span className="text-[10px] font-mono uppercase text-[#64748b] tracking-wider block mb-1">
            Indexed Literature
          </span>
          <div className="text-2xl font-bold text-indigo-400 font-mono">
            {status.indexed_papers_count}
          </div>
          <span className="text-[11px] text-[#94a3b8] mt-0.5 block">
            Scientific research papers
          </span>
        </div>
        <div className="w-10 h-10 rounded-lg bg-indigo-950/40 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
          <BookOpen className="w-5 h-5" />
        </div>
      </div>

      {/* Retrieval Subsystems Card */}
      <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] shadow-lg flex items-center justify-between">
        <div>
          <span className="text-[10px] font-mono uppercase text-[#64748b] tracking-wider block mb-1">
            Retrieval Subsystems
          </span>
          <div className="text-sm font-bold text-violet-400 font-mono flex items-center gap-1.5 mt-1">
            Dense + BM25
          </div>
          <span className="text-[11px] text-[#94a3b8] mt-0.5 block">
            RRF Fusion (k=60) & Intent Routing
          </span>
        </div>
        <div className="w-10 h-10 rounded-lg bg-violet-950/40 border border-violet-700/50 flex items-center justify-center text-violet-400">
          <Layers className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
};
