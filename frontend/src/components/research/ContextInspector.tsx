'use client';

import React from 'react';
import { ChevronDown, ChevronUp, Layers, Copy, Check } from 'lucide-react';
import { RetrievedChunk } from '@/lib/types';

interface ContextInspectorProps {
  chunks: RetrievedChunk[];
}

export const ContextInspector: React.FC<ContextInspectorProps> = ({ chunks }) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const [copiedId, setCopiedId] = React.useState<string | null>(null);

  if (!chunks || chunks.length === 0) return null;

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="border border-[#232732] rounded-xl bg-[#111318] overflow-hidden transition">
      {/* Accordion Toggle Bar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-[#171a21] transition cursor-pointer"
      >
        <div className="flex items-center gap-2.5">
          <Layers className="w-4 h-4 text-indigo-400" />
          <div>
            <h4 className="text-xs font-semibold text-white">
              Inspect Retrieved Context Passages ({chunks.length})
            </h4>
            <p className="text-[11px] text-[#94a3b8]">
              Raw passages retrieved via Hybrid RRF and provided to the synthesizer
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[#64748b]">
            {isOpen ? 'Collapse' : 'Expand'}
          </span>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-[#94a3b8]" />
          ) : (
            <ChevronDown className="w-4 h-4 text-[#94a3b8]" />
          )}
        </div>
      </button>

      {/* Expanded Chunks Content */}
      {isOpen && (
        <div className="p-4 border-t border-[#232732] space-y-3 bg-[#090a0d]/50 animate-fade-in">
          {chunks.map((c, idx) => (
            <div
              key={c.chunk_id || idx}
              className="p-3 rounded-lg bg-[#111318] border border-[#232732] space-y-2"
            >
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-300 font-mono text-[10px] flex items-center justify-center font-bold">
                    #{idx + 1}
                  </span>
                  <span className="font-mono text-xs font-semibold text-white">
                    {c.chunk_id}
                  </span>
                  <span className="text-[10px] text-[#94a3b8]">
                    ({c.document_id} • Page {c.physical_page_number})
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-emerald-400 font-medium">
                    Score: {c.score.toFixed(4)}
                  </span>
                  <button
                    onClick={() => handleCopy(c.chunk_id, c.text)}
                    className="p-1 rounded text-[#94a3b8] hover:text-white hover:bg-[#171a21] transition cursor-pointer"
                    title="Copy passage text"
                  >
                    {copiedId === c.chunk_id ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              </div>

              <p className="text-xs text-[#cbd5e1] leading-relaxed whitespace-pre-wrap font-sans bg-[#090a0d] p-2.5 rounded border border-[#232732]/60">
                {c.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
