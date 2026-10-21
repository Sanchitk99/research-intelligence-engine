'use client';

import React from 'react';
import { Cpu, CheckCircle2, XCircle } from 'lucide-react';
import { ProviderStatus } from '@/lib/types';

interface ProviderGridProps {
  providers: Record<string, ProviderStatus>;
}

export const ProviderGrid: React.FC<ProviderGridProps> = ({ providers }) => {
  const providerMeta: Record<string, { name: string; description: string; iconColor: string }> = {
    mock: {
      name: 'Deterministic Mock Provider',
      description: 'Zero-cost offline evaluator simulating semantic synthesis with 100% verified citations.',
      iconColor: 'text-indigo-400',
    },
    gemini: {
      name: 'Google Gemini',
      description: 'High-throughput multimodal cloud provider with JSON schema validation.',
      iconColor: 'text-blue-400',
    },
    openai: {
      name: 'OpenAI (GPT-4o)',
      description: 'Advanced reasoning and synthesis with structured output constraints.',
      iconColor: 'text-emerald-400',
    },
    groq: {
      name: 'Groq Cloud (Llama 3.3)',
      description: 'Ultra-low latency LPU inference with strict citation extraction.',
      iconColor: 'text-amber-400',
    },
    ollama: {
      name: 'Ollama (Local LLM)',
      description: 'Air-gapped on-device inference using local open weights (Mistral / Gemma).',
      iconColor: 'text-purple-400',
    },
  };

  return (
    <div className="border border-[#232732] rounded-xl bg-[#111318] overflow-hidden shadow-lg p-5 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[#232732]">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            Synthesizer Provider Readiness
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Configured provider backends and default models (credentials remain strictly server-side)
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {Object.entries(providers).map(([key, info]) => {
          const meta = providerMeta[key] || {
            name: key.toUpperCase(),
            description: 'Configured LLM backend service.',
            iconColor: 'text-indigo-400',
          };

          return (
            <div
              key={key}
              className="p-3.5 rounded-lg bg-[#090a0d]/60 border border-[#232732] flex flex-col justify-between gap-3 group"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <h4 className="text-xs font-semibold text-white flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${info.configured ? 'bg-emerald-400' : 'bg-slate-500'}`} />
                    {meta.name}
                  </h4>
                  {info.configured ? (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40">
                      <CheckCircle2 className="w-3 h-3" />
                      Ready
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-[#64748b] bg-[#171a21] px-2 py-0.5 rounded border border-[#232732]">
                      <XCircle className="w-3 h-3" />
                      Not Set
                    </span>
                  )}
                </div>

                <p className="text-[11px] text-[#94a3b8] leading-relaxed">
                  {meta.description}
                </p>
              </div>

              <div className="pt-2 border-t border-[#232732]/60 flex items-center justify-between text-[10px] font-mono">
                <span className="text-[#64748b]">Default Model:</span>
                <span className="text-indigo-300 bg-[#171a21] px-1.5 py-0.5 rounded border border-[#232732]">
                  {info.default_model || 'N/A'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
