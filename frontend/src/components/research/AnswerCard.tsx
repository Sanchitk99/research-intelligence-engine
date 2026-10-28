'use client';

import React from 'react';
import {
  ShieldCheck,
  Clock,
  Cpu,
  Layers,
  Copy,
  Check,
  RefreshCw,
  Info,
  BookOpen,
} from 'lucide-react';
import { AskQuestionResponse } from '@/lib/types';
import { getStatusBadgeConfig, formatLatency } from '@/lib/utils';
import { CitationCard } from './CitationCard';
import { ContextInspector } from './ContextInspector';

interface AnswerCardProps {
  data: AskQuestionResponse;
  onRetry?: (question: string) => void;
}

export const AnswerCard: React.FC<AnswerCardProps> = ({ data, onRetry }) => {
  const [copied, setCopied] = React.useState(false);
  const statusConfig = getStatusBadgeConfig(data.status);

  const handleCopyAnswer = () => {
    navigator.clipboard.writeText(data.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isAbstained = data.status.toLowerCase().includes('abstained');

  return (
    <div className="max-w-4xl mx-auto rounded-xl bg-[#111318] border border-[#232732] shadow-xl overflow-hidden space-y-6 p-6 animate-fade-in">
      {/* Top Question & Status Badge */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-4 border-b border-[#232732]">
        <div className="space-y-1">
          <span className="text-[10px] font-mono uppercase text-[#64748b] tracking-wider block">
            Research Inquiry
          </span>
          <h2 className="text-base sm:text-lg font-bold text-white tracking-tight leading-snug">
            {data.question}
          </h2>
        </div>

        {/* Status Pill */}
        <div className="shrink-0 flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${statusConfig.bg} ${statusConfig.text} ${statusConfig.border}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${statusConfig.dot}`} />
            {statusConfig.label}
          </span>
        </div>
      </div>

      {/* Answer Body */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8] flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Synthesized Evidence
          </span>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyAnswer}
              className="flex items-center gap-1 px-2.5 py-1 rounded text-xs text-[#94a3b8] hover:text-white bg-[#171a21] border border-[#232732] hover:border-[#353c4b] transition cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>

            {onRetry && (
              <button
                onClick={() => onRetry(data.question)}
                className="flex items-center gap-1 px-2.5 py-1 rounded text-xs text-[#94a3b8] hover:text-white bg-[#171a21] border border-[#232732] hover:border-[#353c4b] transition cursor-pointer"
                title="Rerun query"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry</span>
              </button>
            )}
          </div>
        </div>

        {/* Answer Typography */}
        <div className="text-sm text-[#e2e8f0] leading-relaxed whitespace-pre-line bg-[#090a0d]/70 p-5 rounded-lg border border-[#232732]/70 font-sans shadow-inner">
          {data.answer}
        </div>
      </div>

      {/* Confidence Notes Banner if present */}
      {data.confidence_notes && (
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-indigo-950/20 border border-indigo-900/30 text-xs text-indigo-300">
          <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          <div>
            <strong className="font-semibold block text-indigo-200">Confidence & Synthesis Notes:</strong>
            <p className="mt-0.5 text-indigo-300/90 leading-relaxed">{data.confidence_notes}</p>
          </div>
        </div>
      )}

      {/* Verified Citations Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8] flex items-center gap-1.5">
            <BookOpen className="w-4 h-4 text-violet-400" />
            Verified Citations ({data.citations.length})
          </span>
          <span className="text-[11px] text-[#64748b]">
            Every excerpt is matched verbatim in retrieved source chunks
          </span>
        </div>

        {data.citations.length === 0 ? (
          <div className="p-4 rounded-lg bg-[#090a0d]/50 border border-[#232732] text-xs text-[#64748b] text-center">
            {isAbstained
              ? 'Zero citations emitted on abstention (safe fallback enforced; zero hallucinated sources).'
              : 'No citations emitted for this answer.'}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {data.citations.map((c, idx) => (
              <CitationCard key={c.chunk_id || idx} citation={c} index={idx} />
            ))}
          </div>
        )}
      </div>

      {/* Retrieved Context Inspector (Collapsible Accordion) */}
      <ContextInspector chunks={data.retrieved_chunks} />

      {/* Performance & Execution Diagnostics Bar */}
      <div className="pt-4 border-t border-[#232732] flex flex-wrap items-center justify-between gap-3 text-[11px] text-[#94a3b8]">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-[#171a21] border border-[#232732]">
            <Cpu className="w-3 h-3 text-indigo-400" />
            Provider: <strong className="text-white capitalize">{data.provider_name}</strong>
            {data.failover_occurred && (
              <span
                className="ml-1 text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/30 font-medium"
                title={data.primary_provider ? `Failover from ${data.primary_provider}` : "Automatic failover occurred"}
              >
                Failover
              </span>
            )}
          </span>
          <span className="flex items-center gap-1 px-2.5 py-0.5 rounded bg-[#171a21] border border-[#232732]">
            Model: <strong className="text-white font-mono">{data.model_name}</strong>
          </span>
          <span className="flex items-center gap-1 px-2.5 py-0.5 rounded bg-[#171a21] border border-[#232732]">
            <Layers className="w-3 h-3 text-violet-400" />
            Mode: <strong className="text-white uppercase font-mono">{data.retrieval_mode}</strong>
          </span>
        </div>

        <div className="flex items-center gap-3">
          {Boolean(data.total_tokens) && (
            <span className="text-[#64748b] font-mono">
              Tokens: Prompt {data.prompt_tokens || 0} / Comp {data.completion_tokens || 0} ({data.total_tokens})
            </span>
          )}
          <span className="flex items-center gap-1 px-2.5 py-0.5 rounded bg-[#171a21] border border-[#232732] font-mono text-emerald-400 font-medium">
            <Clock className="w-3 h-3" />
            {formatLatency(data.latency_ms)}
          </span>
        </div>
      </div>
    </div>
  );
};
