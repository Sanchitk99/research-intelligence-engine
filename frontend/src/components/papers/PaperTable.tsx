'use client';

import React from 'react';
import { BookOpen, CheckCircle2 } from 'lucide-react';
import { PaperSummary } from '@/lib/types';

interface PaperTableProps {
  papers: PaperSummary[];
  isLoading?: boolean;
}

export const PaperTable: React.FC<PaperTableProps> = ({ papers, isLoading }) => {
  if (isLoading) {
    return (
      <div className="p-8 text-center text-xs text-[#64748b] bg-[#111318] rounded-xl border border-[#232732]">
        Loading indexed papers repository...
      </div>
    );
  }

  if (!papers || papers.length === 0) {
    return (
      <div className="p-8 text-center text-xs text-[#64748b] bg-[#111318] rounded-xl border border-[#232732]">
        No research papers indexed in corpus yet.
      </div>
    );
  }

  return (
    <div className="border border-[#232732] rounded-xl bg-[#111318] overflow-hidden shadow-lg">
      <div className="px-5 py-3.5 border-b border-[#232732] bg-[#171a21] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-violet-400" />
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
            Indexed Literature Corpus ({papers.length} Documents)
          </h3>
        </div>
        <span className="text-[11px] text-[#94a3b8]">
          Total Chunks: <strong>{papers.reduce((sum, p) => sum + p.total_chunks, 0)}</strong>
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[#232732] bg-[#090a0d]/50 text-[#64748b] text-[11px] uppercase font-mono">
              <th className="px-5 py-2.5 font-semibold">ID</th>
              <th className="px-5 py-2.5 font-semibold">Paper Title</th>
              <th className="px-5 py-2.5 font-semibold text-center">Pages</th>
              <th className="px-5 py-2.5 font-semibold text-center">Chunks</th>
              <th className="px-5 py-2.5 font-semibold text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#232732]">
            {papers.map((p) => (
              <tr key={p.document_id} className="hover:bg-[#171a21]/60 transition">
                <td className="px-5 py-3 font-mono font-semibold text-indigo-400">
                  <span className="px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40">
                    {p.document_id}
                  </span>
                </td>
                <td className="px-5 py-3 font-medium text-white max-w-md">
                  {p.title}
                </td>
                <td className="px-5 py-3 text-center text-[#94a3b8] font-mono">
                  {p.total_pages}
                </td>
                <td className="px-5 py-3 text-center font-mono font-semibold text-indigo-300">
                  {p.total_chunks}
                </td>
                <td className="px-5 py-3 text-right">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-950/40 text-emerald-300 border border-emerald-700/50">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    Indexed
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
