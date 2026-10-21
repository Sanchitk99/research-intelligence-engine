'use client';

import React from 'react';
import { Play, Eye, Sparkles } from 'lucide-react';
import { CorpusPDF } from '@/lib/types';
import { formatBytes } from '@/lib/utils';

interface CorpusListProps {
  corpusPDFs: CorpusPDF[];
  onIngest: (filename: string, dryRun: boolean) => void;
  isIngesting: boolean;
}

export const CorpusList: React.FC<CorpusListProps> = ({
  corpusPDFs,
  onIngest,
  isIngesting,
}) => {
  return (
    <div className="border border-[#232732] rounded-xl bg-[#111318] overflow-hidden shadow-lg space-y-4 p-5">
      <div className="flex items-center justify-between pb-3 border-b border-[#232732]">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            Benchmark Evaluation Corpus (PDF Documents)
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Pre-bundled scientific papers available in <code className="text-indigo-300">evaluation/corpus/papers/</code>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {corpusPDFs.map((pdf) => (
          <div
            key={pdf.filename}
            className="p-3.5 rounded-lg bg-[#090a0d]/60 border border-[#232732] hover:border-indigo-500/40 transition flex flex-col justify-between gap-3 group"
          >
            <div>
              <div className="flex items-start justify-between gap-2 mb-1">
                <span className="font-mono text-xs font-bold text-indigo-400 px-1.5 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40 shrink-0">
                  {pdf.document_id}
                </span>
                <span className="text-[10px] font-mono text-[#64748b]">
                  {formatBytes(pdf.file_size_bytes)}
                </span>
              </div>
              <h4 className="text-xs font-semibold text-white group-hover:text-indigo-300 transition line-clamp-2">
                {pdf.title}
              </h4>
              <p className="text-[10px] text-[#64748b] font-mono mt-1 truncate">
                {pdf.filename}
              </p>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#232732]/60">
              <button
                disabled={isIngesting}
                onClick={() => onIngest(pdf.filename, true)}
                className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-medium text-[#94a3b8] hover:text-white bg-[#171a21] border border-[#232732] hover:border-[#3b4455] transition cursor-pointer disabled:opacity-50"
                title="Extract and chunk without persisting to Qdrant"
              >
                <Eye className="w-3 h-3 text-indigo-400" />
                <span>Dry Run</span>
              </button>

              <button
                disabled={isIngesting}
                onClick={() => onIngest(pdf.filename, false)}
                className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-medium text-white bg-indigo-600 hover:bg-indigo-500 transition cursor-pointer disabled:opacity-50 shadow-sm"
                title="Ingest, chunk, embed, and index into Qdrant"
              >
                <Play className="w-3 h-3 fill-current" />
                <span>Ingest & Index</span>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
