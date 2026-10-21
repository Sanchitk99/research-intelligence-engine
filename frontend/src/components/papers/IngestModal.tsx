'use client';

import React from 'react';
import { X, Upload, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { IngestionResponse } from '@/lib/types';

interface IngestModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (file: File, dryRun: boolean, docId?: string, docTitle?: string) => Promise<IngestionResponse>;
}

export const IngestModal: React.FC<IngestModalProps> = ({ isOpen, onClose, onUpload }) => {
  const [file, setFile] = React.useState<File | null>(null);
  const [docId, setDocId] = React.useState('');
  const [docTitle, setDocTitle] = React.useState('');
  const [dryRun, setDryRun] = React.useState(true);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [report, setReport] = React.useState<IngestionResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (!selected.name.toLowerCase().endsWith('.pdf')) {
        setError('Only .pdf files are supported.');
        return;
      }
      setFile(selected);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsSubmitting(true);
    setError(null);
    setReport(null);

    try {
      const result = await onUpload(file, dryRun, docId || undefined, docTitle || undefined);
      setReport(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Ingestion failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-lg bg-[#111318] border border-[#232732] rounded-xl shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-[#232732] flex items-center justify-between bg-[#171a21]">
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">Ingest & Chunk Research Paper</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-[#94a3b8] hover:text-white hover:bg-[#232732] transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 space-y-4 max-h-[80vh] overflow-y-auto">
          {!report ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* File Dropzone */}
              <div>
                <label className="block text-xs font-medium text-[#94a3b8] mb-1.5">
                  Target Research PDF
                </label>
                <div className="border-2 border-dashed border-[#232732] hover:border-indigo-500/50 rounded-xl p-6 text-center bg-[#090a0d]/40 transition cursor-pointer relative">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                  />
                  <div className="flex flex-col items-center gap-2">
                    <FileText className="w-8 h-8 text-indigo-400/80" />
                    {file ? (
                      <div>
                        <p className="text-xs font-semibold text-white">{file.name}</p>
                        <p className="text-[11px] text-[#64748b]">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-xs font-medium text-white">Click or drag & drop PDF here</p>
                        <p className="text-[11px] text-[#64748b]">Maximum file size: 25MB</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Optional Metadata Overrides */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-[#94a3b8] mb-1">
                    Document ID (Optional)
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. P6"
                    value={docId}
                    onChange={(e) => setDocId(e.target.value)}
                    className="w-full px-3 py-2 bg-[#090a0d] border border-[#232732] rounded-md text-xs text-white placeholder-[#64748b] focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-[#94a3b8] mb-1">
                    Paper Title (Optional)
                  </label>
                  <input
                    type="text"
                    placeholder="Auto-detected from PDF"
                    value={docTitle}
                    onChange={(e) => setDocTitle(e.target.value)}
                    className="w-full px-3 py-2 bg-[#090a0d] border border-[#232732] rounded-md text-xs text-white placeholder-[#64748b] focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              {/* Dry Run Checkbox */}
              <div className="p-3 rounded-lg bg-[#171a21] border border-[#232732]">
                <label className="flex items-start gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={dryRun}
                    onChange={(e) => setDryRun(e.target.checked)}
                    className="mt-0.5 rounded border-[#3b4455] text-indigo-600 focus:ring-indigo-500"
                  />
                  <div>
                    <span className="text-xs font-medium text-white">
                      Dry Run Mode (Safe Preview)
                    </span>
                    <p className="text-[11px] text-[#94a3b8] mt-0.5">
                      Extracts pages and generates chunks without modifying or persisting to Qdrant vector storage.
                    </p>
                  </div>
                </label>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/50 text-xs text-rose-300 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

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
                  disabled={!file || isSubmitting}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:bg-[#171a21] text-white disabled:text-[#64748b] text-xs font-medium transition cursor-pointer disabled:cursor-not-allowed shadow-sm"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Processing PDF...</span>
                    </>
                  ) : (
                    <span>{dryRun ? 'Execute Dry Run' : 'Ingest & Index'}</span>
                  )}
                </button>
              </div>
            </form>
          ) : (
            /* Ingestion Report Card */
            <div className="space-y-4 animate-fade-in">
              <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-300 flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="block text-white font-semibold">
                    {report.success ? 'Document Processed Successfully' : 'Ingestion Failed'}
                  </strong>
                  <p className="text-[11px] text-emerald-300/90 mt-0.5">
                    {report.title} ({report.document_id})
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                <div className="p-2.5 rounded-lg bg-[#090a0d] border border-[#232732]">
                  <span className="text-[10px] text-[#64748b] uppercase block">Pages Read</span>
                  <span className="text-sm font-bold text-white font-mono">{report.pages_extracted}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#090a0d] border border-[#232732]">
                  <span className="text-[10px] text-[#64748b] uppercase block">Chunks Created</span>
                  <span className="text-sm font-bold text-indigo-400 font-mono">{report.chunks_created}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#090a0d] border border-[#232732]">
                  <span className="text-[10px] text-[#64748b] uppercase block">Chunks Indexed</span>
                  <span className="text-sm font-bold text-emerald-400 font-mono">{report.chunks_indexed}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-[#090a0d] border border-[#232732]">
                  <span className="text-[10px] text-[#64748b] uppercase block">Total Points</span>
                  <span className="text-sm font-bold text-violet-400 font-mono">{report.total_collection_points}</span>
                </div>
              </div>

              {/* Sample Chunks Previews */}
              {report.sample_chunks && report.sample_chunks.length > 0 && (
                <div className="space-y-2">
                  <span className="text-[11px] font-semibold text-[#94a3b8] uppercase tracking-wider block">
                    Sample Chunk Previews
                  </span>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {report.sample_chunks.map((sc, i) => (
                      <div key={i} className="p-2 rounded bg-[#090a0d] border border-[#232732] text-[11px] text-[#cbd5e1] font-mono">
                        <strong className="text-indigo-400">{sc.chunk_id || `Chunk #${i + 1}`}:</strong>{' '}
                        {sc.preview || JSON.stringify(sc)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-2 flex justify-end border-t border-[#232732]">
                <button
                  type="button"
                  onClick={() => {
                    setReport(null);
                    setFile(null);
                    onClose();
                  }}
                  className="px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition cursor-pointer"
                >
                  Done
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
