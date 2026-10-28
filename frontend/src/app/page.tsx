'use client';

import React from 'react';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { SettingsModal } from '@/components/layout/SettingsModal';
import { QueryComposer } from '@/components/research/QueryComposer';
import { AnswerCard } from '@/components/research/AnswerCard';
import { EmptyWorkspace } from '@/components/research/EmptyWorkspace';
import { LoadingSkeleton } from '@/components/research/LoadingSkeleton';
import { PaperTable } from '@/components/papers/PaperTable';
import { CorpusList } from '@/components/papers/CorpusList';
import { IngestModal } from '@/components/papers/IngestModal';
import { SystemOverview } from '@/components/system/SystemOverview';
import { ProviderGrid } from '@/components/system/ProviderGrid';
import { ArchitectureCard } from '@/components/system/ArchitectureCard';
import { ProjectOverview } from '@/components/about/ProjectOverview';

import { api } from '@/lib/api';
import {
  AskQuestionResponse,
  EngineSettings,
  PaperSummary,
  CorpusPDF,
  SystemStatus,
  ResearchSession,
  IngestionResponse,
} from '@/lib/types';
import { AlertCircle, Upload, RefreshCw } from 'lucide-react';

export default function Home() {
  // Navigation & Modal State
  const [currentTab, setCurrentTab] = React.useState<'research' | 'papers' | 'status' | 'about'>('research');
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
  const [isIngestModalOpen, setIsIngestModalOpen] = React.useState(false);

  // Engine Settings
  const [settings, setSettings] = React.useState<EngineSettings>({
    provider: 'mock',
    model: '',
    retrieval_mode: 'hybrid',
    top_k: 5,
    enable_intent_routing: false,
  });

  // Research State
  const [sessions, setSessions] = React.useState<ResearchSession[]>([]);
  const [activeSessionId, setActiveSessionId] = React.useState<string | null>(null);
  const [composerQuery, setComposerQuery] = React.useState<string>('');
  const [currentAnswer, setCurrentAnswer] = React.useState<AskQuestionResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [queryError, setQueryError] = React.useState<string | null>(null);

  // Papers & System Data State
  const [papers, setPapers] = React.useState<PaperSummary[]>([]);
  const [corpusPDFs, setCorpusPDFs] = React.useState<CorpusPDF[]>([]);
  const [systemStatus, setSystemStatus] = React.useState<SystemStatus | null>(null);
  const [isDataLoading, setIsDataLoading] = React.useState(false);
  const [ingestNotice, setIngestNotice] = React.useState<string | null>(null);

  // Load initial settings and sessions from localStorage
  React.useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('rie_settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }

      const savedSessions = localStorage.getItem('rie_sessions');
      if (savedSessions) {
        const parsed = JSON.parse(savedSessions);
        setSessions(parsed);
        // Do not auto-restore past queries into the active viewport on initial mount.
        // A fresh session starts in the curated EmptyWorkspace with benchmark prompts.
        // Users can review previous inquiries by explicitly clicking them in the sidebar.
        setActiveSessionId(null);
        setCurrentAnswer(null);
      }
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Fetch papers and system status
  const loadSystemData = React.useCallback(async () => {
    setIsDataLoading(true);
    try {
      const [statusRes, papersRes, corpusRes] = await Promise.allSettled([
        api.getSystemStatus(),
        api.getIndexedPapers(),
        api.getCorpusPDFs(),
      ]);

      if (statusRes.status === 'fulfilled') setSystemStatus(statusRes.value);
      if (papersRes.status === 'fulfilled') setPapers(papersRes.value);
      if (corpusRes.status === 'fulfilled') setCorpusPDFs(corpusRes.value);
    } catch (err) {
      console.error('Failed to load system data:', err);
    } finally {
      setIsDataLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadSystemData();
  }, [loadSystemData]);

  // Save settings handler
  const handleSaveSettings = (newSettings: EngineSettings) => {
    setSettings(newSettings);
    try {
      localStorage.setItem('rie_settings', JSON.stringify(newSettings));
    } catch {
      // Ignore localStorage errors
    }
  };

  // Start a new research session
  const handleNewSession = () => {
    setActiveSessionId(null);
    setCurrentAnswer(null);
    setComposerQuery('');
    setQueryError(null);
  };

  // Select existing session from sidebar
  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
    const session = sessions.find((s) => s.id === id);
    if (session && session.history.length > 0) {
      const lastQa = session.history[session.history.length - 1];
      setCurrentAnswer(lastQa.response);
      setComposerQuery(lastQa.question);
    } else {
      setCurrentAnswer(null);
      setComposerQuery('');
    }
    setQueryError(null);
  };

  // Delete a session
  const handleDeleteSession = (id: string) => {
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id) {
      setActiveSessionId(null);
      setCurrentAnswer(null);
      setComposerQuery('');
    }
    try {
      localStorage.setItem('rie_sessions', JSON.stringify(updated));
    } catch {
      // Ignore
    }
  };

  // Execute Question Search
  const handleSearch = async (question: string) => {
    setIsLoading(true);
    setQueryError(null);
    setComposerQuery(question);

    try {
      const response = await api.askQuestion({
        question,
        provider: settings.provider,
        model: settings.model.trim() ? settings.model.trim() : null,
        retrieval_mode: settings.retrieval_mode,
        top_k: settings.top_k,
        enable_intent_routing: settings.enable_intent_routing,
      });

      setCurrentAnswer(response);

      // Append to session history
      const currentSessionId = activeSessionId || `session_${Date.now()}`;
      const newQaPair = {
        id: `qa_${Date.now()}`,
        question,
        timestamp: new Date().toISOString(),
        response,
      };

      setSessions((prev) => {
        const existing = prev.find((s) => s.id === currentSessionId);
        let updated: ResearchSession[];
        if (existing) {
          updated = prev.map((s) =>
            s.id === currentSessionId
              ? {
                  ...s,
                  title: s.history.length === 0 ? question.slice(0, 45) : s.title,
                  history: [...s.history, newQaPair],
                }
              : s
          );
        } else {
          updated = [
            {
              id: currentSessionId,
              title: question.slice(0, 45),
              createdAt: new Date().toISOString(),
              history: [newQaPair],
            },
            ...prev,
          ];
        }
        try {
          localStorage.setItem('rie_sessions', JSON.stringify(updated));
        } catch {
          // Ignore
        }
        return updated;
      });

      if (!activeSessionId) {
        setActiveSessionId(currentSessionId);
      }
    } catch (err: unknown) {
      setQueryError(err instanceof Error ? err.message : 'An error occurred during synthesis.');
    } finally {
      setIsLoading(false);
    }
  };

  // Select curated question from EmptyWorkspace
  const handleSelectCuratedInquiry = (question: string) => {
    handleSearch(question);
  };

  // Ingest Corpus Paper Handler
  const handleIngestCorpus = async (filename: string, dryRun: boolean) => {
    setIsDataLoading(true);
    setIngestNotice(null);
    try {
      const report: IngestionResponse = await api.ingestCorpusPaper(filename, dryRun);
      if (report.success) {
        setIngestNotice(
          `${dryRun ? 'Dry run completed' : 'Ingestion succeeded'}: Processed ${report.pages_extracted} pages (${report.chunks_created} chunks created) for ${report.document_id}.`
        );
        await loadSystemData();
      } else {
        setIngestNotice(`Ingestion error: ${report.error_message || 'Unknown error'}`);
      }
    } catch (err: unknown) {
      setIngestNotice(`Ingestion error: ${err instanceof Error ? err.message : 'Failed'}`);
    } finally {
      setIsDataLoading(false);
    }
  };

  // Upload Custom PDF Handler
  const handleUploadPaper = async (
    file: File,
    dryRun: boolean,
    docId?: string,
    docTitle?: string
  ): Promise<IngestionResponse> => {
    const report = await api.uploadAndIngestPDF(file, dryRun, docId, docTitle);
    if (report.success && !dryRun) {
      await loadSystemData();
    }
    return report;
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#090a0d] text-[#f8fafc]">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onNewSession={handleNewSession}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        providerName={settings.provider}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Global Header */}
        <Header
          currentTab={currentTab}
          settings={settings}
          onOpenSettings={() => setIsSettingsOpen(true)}
          totalPoints={systemStatus?.total_points}
        />

        {/* Scrollable Tab Content Body */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 max-w-6xl mx-auto w-full">
          {/* TAB 1: RESEARCH WORKSPACE */}
          {currentTab === 'research' && (
            <div className="space-y-6">
              {/* Question Composer */}
              <QueryComposer
                onSearch={handleSearch}
                isLoading={isLoading}
                initialValue={composerQuery}
              />

              {/* Error Banner */}
              {queryError && (
                <div className="max-w-4xl mx-auto p-4 rounded-xl bg-rose-950/40 border border-rose-800/50 text-xs text-rose-300 flex items-start gap-2.5 animate-fade-in shadow-lg">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="block text-white font-semibold">Synthesis Error</strong>
                    <p className="mt-0.5 text-rose-200/90">{queryError}</p>
                    <p className="mt-1 text-[11px] text-rose-400/80">
                      Verify backend connectivity or switch to the offline Mock Provider in Settings.
                    </p>
                  </div>
                </div>
              )}

              {/* Loading State Skeleton */}
              {isLoading && <LoadingSkeleton />}

              {/* Render Current Answer Card */}
              {!isLoading && currentAnswer && (
                <AnswerCard data={currentAnswer} onRetry={handleSearch} />
              )}

              {/* Curated Empty Workspace if no answer yet */}
              {!isLoading && !currentAnswer && !queryError && (
                <EmptyWorkspace
                  onSelectQuery={handleSelectCuratedInquiry}
                  isLoading={isLoading}
                />
              )}
            </div>
          )}

          {/* TAB 2: PAPER LIBRARY */}
          {currentTab === 'papers' && (
            <div className="space-y-6">
              {/* Action Bar */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold text-white">Literature Corpus & Ingestion</h3>
                  <p className="text-xs text-[#94a3b8]">
                    Manage indexed scientific publications and upload custom PDF papers
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={loadSystemData}
                    disabled={isDataLoading}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#111318] hover:bg-[#171a21] text-[#94a3b8] hover:text-white border border-[#232732] text-xs font-medium transition cursor-pointer"
                    title="Refresh paper index"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isDataLoading ? 'animate-spin' : ''}`} />
                    <span>Refresh</span>
                  </button>

                  <button
                    onClick={() => setIsIngestModalOpen(true)}
                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition cursor-pointer shadow-sm shadow-indigo-600/20"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload & Ingest PDF</span>
                  </button>
                </div>
              </div>

              {/* Ingest Notice */}
              {ingestNotice && (
                <div className="p-3.5 rounded-xl bg-indigo-950/40 border border-indigo-800/50 text-xs text-indigo-300 flex items-center justify-between animate-fade-in">
                  <span>{ingestNotice}</span>
                  <button
                    onClick={() => setIngestNotice(null)}
                    className="text-[#94a3b8] hover:text-white text-xs font-mono ml-3 cursor-pointer"
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {/* Indexed Papers Table */}
              <PaperTable papers={papers} isLoading={isDataLoading} />

              {/* Available Benchmark Corpus List */}
              <CorpusList
                corpusPDFs={corpusPDFs}
                onIngest={handleIngestCorpus}
                isIngesting={isDataLoading}
              />
            </div>
          )}

          {/* TAB 3: SYSTEM STATUS */}
          {currentTab === 'status' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-semibold text-white">System Architecture & Health</h3>
                <p className="text-xs text-[#94a3b8]">
                  Live vector index telemetry, active collection points, and provider configuration status
                </p>
              </div>

              {/* Metrics Overview */}
              <SystemOverview status={systemStatus} isLoading={isDataLoading} />

              {/* Provider Grid */}
              {systemStatus && <ProviderGrid providers={systemStatus.providers} />}

              {/* Architectural Pipeline Overview */}
              <ArchitectureCard />
            </div>
          )}

          {/* TAB 4: ABOUT THIS PROJECT */}
          {currentTab === 'about' && (
            <ProjectOverview totalPoints={systemStatus?.total_points} />
          )}
        </main>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={handleSaveSettings}
      />

      {/* Ingestion & Upload Modal */}
      <IngestModal
        isOpen={isIngestModalOpen}
        onClose={() => {
          setIsIngestModalOpen(false);
          loadSystemData();
        }}
        onUpload={handleUploadPaper}
      />
    </div>
  );
}
