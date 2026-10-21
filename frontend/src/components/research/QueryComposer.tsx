'use client';

import React from 'react';
import { Send, Loader2, X, CornerDownLeft } from 'lucide-react';

interface QueryComposerProps {
  onSearch: (question: string) => void;
  isLoading: boolean;
  initialValue?: string;
}

export const QueryComposer: React.FC<QueryComposerProps> = ({
  onSearch,
  isLoading,
  initialValue = '',
}) => {
  const [query, setQuery] = React.useState(initialValue);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  React.useEffect(() => {
    if (initialValue) {
      setQuery(initialValue);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [initialValue]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSearch(query.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClear = () => {
    setQuery('');
    if (textareaRef.current) textareaRef.current.focus();
  };

  return (
    <div className="w-full space-y-2 max-w-4xl mx-auto">
      <form
        onSubmit={handleSubmit}
        className="relative bg-[#111318] border border-[#232732] focus-within:border-indigo-500/60 focus-within:ring-1 focus-within:ring-indigo-500/30 rounded-xl shadow-lg transition duration-150 overflow-hidden"
      >
        <div className="relative">
          <textarea
            ref={textareaRef}
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a technical research question across the indexed scientific literature..."
            disabled={isLoading}
            className="w-full bg-transparent px-4 py-3.5 text-sm text-white placeholder-[#64748b] focus:outline-none resize-none leading-relaxed"
          />

          {query && !isLoading && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute top-3.5 right-3 p-1 rounded text-[#64748b] hover:text-[#94a3b8] transition cursor-pointer"
              title="Clear question"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Composer Action Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-[#232732] bg-[#090a0d]/60">
          <div className="flex items-center gap-2 text-[11px] text-[#64748b]">
            <span className="hidden sm:inline-flex items-center gap-1 font-mono">
              <CornerDownLeft className="w-3 h-3 text-[#94a3b8]" />
              <kbd className="px-1.5 py-0.5 rounded bg-[#171a21] border border-[#232732] text-[10px] text-[#94a3b8]">
                Enter
              </kbd>
              to submit
            </span>
          </div>

          <button
            type="submit"
            disabled={!query.trim() || isLoading}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-[#171a21] text-white disabled:text-[#64748b] text-xs font-semibold transition cursor-pointer disabled:cursor-not-allowed shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Synthesizing...</span>
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" />
                <span>Generate Answer</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
