'use client';

import React from 'react';
import { X, Sliders, Sparkles } from 'lucide-react';
import { EngineSettings } from '@/lib/types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: EngineSettings;
  onSaveSettings: (settings: EngineSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
}) => {
  const [form, setForm] = React.useState<EngineSettings>(settings);

  React.useEffect(() => {
    setForm(settings);
  }, [settings, isOpen]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveSettings(form);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-md bg-[#111318] border border-[#232732] rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#232732] flex items-center justify-between bg-[#171a21]">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Engine Configuration</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-[#94a3b8] hover:text-white hover:bg-[#232732] transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {/* Provider Selection */}
          <div>
            <label className="block text-xs font-medium text-[#94a3b8] mb-1.5">
              LLM Synthesizer Provider
            </label>
            <select
              value={form.provider}
              onChange={(e) => setForm({ ...form, provider: e.target.value })}
              className="w-full px-3 py-2 bg-[#090a0d] border border-[#232732] rounded-md text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              <option value="auto">Auto (Configured Priority)</option>
              <option value="mock">Mock Provider (Deterministic Offline)</option>
              <option value="gemini">Google Gemini (Gemini-Flash-Lite)</option>
              <option value="openai">OpenAI (GPT-4o-mini)</option>
              <option value="groq">Groq (Llama-3.3-70B)</option>
              <option value="ollama">Ollama (Local mistral / gemma)</option>
            </select>
            <p className="text-[11px] text-[#64748b] mt-1">
              Select &apos;mock&apos; for reproducible, offline demonstration without API keys.
            </p>
          </div>

          {/* Model Override */}
          <div>
            <label className="block text-xs font-medium text-[#94a3b8] mb-1.5">
              Model Identifier Override (Optional)
            </label>
            <input
              type="text"
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              placeholder="e.g. gemini-flash-lite-latest or gpt-4o-mini"
              className="w-full px-3 py-2 bg-[#090a0d] border border-[#232732] rounded-md text-xs text-white placeholder-[#64748b] focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* Retrieval Mode */}
          <div>
            <label className="block text-xs font-medium text-[#94a3b8] mb-1.5">
              Retrieval Strategy
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setForm({ ...form, retrieval_mode: 'hybrid' })}
                className={`px-3 py-2 rounded-md text-xs font-medium border text-center transition cursor-pointer ${
                  form.retrieval_mode === 'hybrid'
                    ? 'bg-indigo-600/20 text-indigo-300 border-indigo-500/50'
                    : 'bg-[#090a0d] text-[#94a3b8] border-[#232732] hover:text-white'
                }`}
              >
                Hybrid (Dense + BM25)
              </button>
              <button
                type="button"
                onClick={() => setForm({ ...form, retrieval_mode: 'dense' })}
                className={`px-3 py-2 rounded-md text-xs font-medium border text-center transition cursor-pointer ${
                  form.retrieval_mode === 'dense'
                    ? 'bg-indigo-600/20 text-indigo-300 border-indigo-500/50'
                    : 'bg-[#090a0d] text-[#94a3b8] border-[#232732] hover:text-white'
                }`}
              >
                Dense Vector Only
              </button>
            </div>
            <p className="text-[11px] text-[#64748b] mt-1">
              Hybrid fuses SBERT embeddings with Okapi BM25 keyword matching via Reciprocal Rank Fusion ($k=60$).
            </p>
          </div>

          {/* Top-K Chunks Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-[#94a3b8]">
                Top-k Context Passages
              </label>
              <span className="text-xs font-mono font-semibold text-white px-2 py-0.5 bg-[#171a21] rounded border border-[#232732]">
                {form.top_k} passages
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={form.top_k}
              onChange={(e) => setForm({ ...form, top_k: parseInt(e.target.value, 10) })}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>

          {/* Intent Routing Toggle */}
          <div className="p-3 rounded-lg bg-[#171a21] border border-[#232732]">
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={form.enable_intent_routing}
                onChange={(e) => setForm({ ...form, enable_intent_routing: e.target.checked })}
                className="mt-0.5 rounded border-[#3b4455] text-indigo-600 focus:ring-indigo-500"
              />
              <div>
                <span className="text-xs font-medium text-white flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                  Query-Intent Routing (Task 4.4.19)
                </span>
                <p className="text-[11px] text-[#94a3b8] mt-0.5 leading-relaxed">
                  Automatically detects comparative questions and diversifies context across papers (improves multi-paper coverage from 55.6% to 100%).
                </p>
              </div>
            </label>
          </div>

          {/* Footer Actions */}
          <div className="pt-2 flex items-center justify-end gap-2 border-t border-[#232732]">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded-md text-xs text-[#94a3b8] hover:text-white transition cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition cursor-pointer shadow-sm"
            >
              Apply Settings
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
