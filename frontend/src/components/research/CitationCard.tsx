'use client';

import React from 'react';
import { Quote, CheckCircle2 } from 'lucide-react';
import { Citation } from '@/lib/types';

interface CitationCardProps {
  citation: Citation;
  index: number;
}

export const CitationCard: React.FC<CitationCardProps> = ({ citation, index }) => {
  return (
    <div
      id={`citation-${citation.chunk_id}`}
      className="p-3.5 rounded-lg bg-[#111318] border border-[#232732] hover:border-indigo-500/40 transition group relative"
    >
      {/* Card Header */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-indigo-400 text-[11px] font-mono font-semibold flex items-center justify-center shrink-0">
            {index + 1}
          </span>
          <div>
            <h4 className="text-xs font-semibold text-white group-hover:text-indigo-300 transition line-clamp-1">
              {citation.document_title}
            </h4>
            <div className="flex items-center gap-2 text-[10px] text-[#94a3b8] mt-0.5">
              <span className="font-mono uppercase font-semibold text-indigo-400 px-1 py-0.2 bg-indigo-950/60 rounded border border-indigo-800/40">
                {citation.document_id}
              </span>
              <span>•</span>
              <span>Physical Page: <strong>{citation.physical_page_number}</strong></span>
              {citation.printed_page_number && (
                <>
                  <span>•</span>
                  <span>Printed p. {citation.printed_page_number}</span>
                </>
              )}
            </div>
          </div>
        </div>

        <span className="text-[10px] font-mono text-[#64748b] px-1.5 py-0.5 rounded bg-[#171a21] border border-[#232732]">
          {citation.chunk_id}
        </span>
      </div>

      {/* Quoted Excerpt */}
      <div className="relative pl-3 border-l-2 border-indigo-500/40 bg-[#090a0d]/50 rounded-r p-2 mt-2">
        <Quote className="w-3 h-3 text-indigo-400/60 absolute -top-1 -left-1.5 bg-[#111318]" />
        <p className="text-[11px] text-[#cbd5e1] italic font-serif leading-relaxed">
          &ldquo;{citation.excerpt}&rdquo;
        </p>
      </div>

      <div className="mt-2 flex items-center justify-between text-[10px] text-[#64748b]">
        <span className="flex items-center gap-1 text-emerald-400/90">
          <CheckCircle2 className="w-3 h-3" />
          Verified against retrieved context
        </span>
      </div>
    </div>
  );
};
