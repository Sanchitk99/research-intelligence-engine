'use client';

import React from 'react';
import { Loader2, Sparkles } from 'lucide-react';

export const LoadingSkeleton: React.FC = () => {
  const [seconds, setSeconds] = React.useState(0);
  const [stepIndex, setStepIndex] = React.useState(0);

  const steps = [
    'Encoding semantic query vectors via SentenceTransformers...',
    'Performing dense retrieval & Okapi BM25 keyword matching...',
    'Fusing candidate chunks using Reciprocal Rank Fusion (k=60)...',
    'Executing evidence-grounded synthesis & citation verification...',
  ];

  React.useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    const stepTimer = setInterval(() => {
      setStepIndex((idx) => (idx + 1) % steps.length);
    }, 2500);

    return () => {
      clearInterval(timer);
      clearInterval(stepTimer);
    };
  }, [steps.length]);

  return (
    <div className="p-6 rounded-xl bg-[#111318] border border-indigo-500/30 shadow-xl space-y-4 animate-fade-in relative overflow-hidden max-w-4xl mx-auto">
      {/* Background glow */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
          <span className="text-xs font-semibold text-white tracking-wide">
            Synthesizing Grounded Research Evidence...
          </span>
        </div>
        <span className="text-xs font-mono text-indigo-300 px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40">
          {seconds}s elapsed
        </span>
      </div>

      {/* Rotating step info */}
      <div className="flex items-center gap-2 text-xs text-[#94a3b8] py-1">
        <Sparkles className="w-3.5 h-3.5 text-violet-400 shrink-0" />
        <span className="font-mono text-[11px]">{steps[stepIndex]}</span>
      </div>

      {/* Shimmer skeleton lines */}
      <div className="space-y-2.5 pt-2">
        <div className="h-4 bg-[#171a21] rounded-md w-full animate-pulse" />
        <div className="h-4 bg-[#171a21] rounded-md w-11/12 animate-pulse" />
        <div className="h-4 bg-[#171a21] rounded-md w-4/5 animate-pulse" />
        <div className="h-4 bg-[#171a21] rounded-md w-2/3 animate-pulse" />
      </div>

      {/* Mock citation cards skeleton */}
      <div className="pt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
        <div className="h-16 bg-[#171a21]/60 rounded-lg border border-[#232732] animate-pulse" />
        <div className="h-16 bg-[#171a21]/60 rounded-lg border border-[#232732] animate-pulse" />
      </div>
    </div>
  );
};
