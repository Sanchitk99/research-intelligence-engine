'use client';

import React from 'react';
import {
  Compass,
  BookOpen,
  Activity,
  PlusCircle,
  Settings as SettingsIcon,
  FileText,
  Clock,
  Sparkles,
  Trash2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { ResearchSession } from '@/lib/types';

interface SidebarProps {
  currentTab: 'research' | 'papers' | 'status';
  onSelectTab: (tab: 'research' | 'papers' | 'status') => void;
  onOpenSettings: () => void;
  onNewSession: () => void;
  sessions: ResearchSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onDeleteSession?: (id: string) => void;
  providerName: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onOpenSettings,
  onNewSession,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  providerName,
}) => {
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  return (
    <aside
      className={`${
        isCollapsed ? 'w-16' : 'w-64'
      } bg-[#111318] border-r border-[#232732] flex flex-col h-screen select-none shrink-0 transition-all duration-200 z-30`}
    >
      {/* Brand & App Identity */}
      <div className="p-4 border-b border-[#232732] flex items-center justify-between">
        {!isCollapsed && (
          <div className="flex items-center gap-2.5 overflow-hidden">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 shrink-0">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div className="truncate">
              <h1 className="font-semibold text-xs tracking-tight text-white flex items-center gap-1.5 truncate">
                ResearchEngine
                <span className="text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-indigo-950/60 text-indigo-400 border border-indigo-800/40">
                  v1.0
                </span>
              </h1>
              <p className="text-[10px] text-[#64748b] truncate">Evidence-Grounded AI</p>
            </div>
          </div>
        )}

        {isCollapsed && (
          <div className="w-8 h-8 mx-auto rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
        )}

        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1 rounded text-[#64748b] hover:text-[#94a3b8] hover:bg-[#171a21] transition cursor-pointer"
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* New Session Button */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className={`w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition shadow-sm active:scale-[0.99] cursor-pointer ${
            isCollapsed ? 'px-0' : ''
          }`}
          title="Start fresh research session"
        >
          <PlusCircle className="w-4 h-4" />
          {!isCollapsed && <span>New Inquiry</span>}
        </button>
      </div>

      {/* Main Navigation */}
      <div className="px-3 py-1">
        {!isCollapsed && (
          <div className="text-[10px] font-semibold tracking-wider text-[#64748b] uppercase px-2 mb-1.5">
            Navigation
          </div>
        )}
        <nav className="space-y-1">
          <button
            onClick={() => onSelectTab('research')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition cursor-pointer ${
              currentTab === 'research'
                ? 'bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-semibold'
                : 'text-[#94a3b8] hover:text-white hover:bg-[#171a21]'
            } ${isCollapsed ? 'justify-center px-0' : ''}`}
            title="Research Workspace"
          >
            <Compass className="w-4 h-4 text-indigo-400 shrink-0" />
            {!isCollapsed && <span>Research Workspace</span>}
          </button>

          <button
            onClick={() => onSelectTab('papers')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition cursor-pointer ${
              currentTab === 'papers'
                ? 'bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-semibold'
                : 'text-[#94a3b8] hover:text-white hover:bg-[#171a21]'
            } ${isCollapsed ? 'justify-center px-0' : ''}`}
            title="Paper Library"
          >
            <BookOpen className="w-4 h-4 text-violet-400 shrink-0" />
            {!isCollapsed && <span>Paper Library</span>}
          </button>

          <button
            onClick={() => onSelectTab('status')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition cursor-pointer ${
              currentTab === 'status'
                ? 'bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-semibold'
                : 'text-[#94a3b8] hover:text-white hover:bg-[#171a21]'
            } ${isCollapsed ? 'justify-center px-0' : ''}`}
            title="System Telemetry"
          >
            <Activity className="w-4 h-4 text-emerald-400 shrink-0" />
            {!isCollapsed && <span>System Telemetry</span>}
          </button>
        </nav>
      </div>

      {/* Recent Sessions List */}
      {!isCollapsed && (
        <div className="flex-1 overflow-y-auto px-3 py-2 mt-2 border-t border-[#232732]/60">
          <div className="text-[10px] font-semibold tracking-wider text-[#64748b] uppercase px-2 mb-1.5 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            Inquiry History
          </div>

          {sessions.length === 0 ? (
            <div className="px-3 py-4 text-center text-xs text-[#64748b]">
              No saved inquiries.
            </div>
          ) : (
            <div className="space-y-1">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`group/item flex items-center justify-between rounded-lg px-2.5 py-1.5 text-xs transition cursor-pointer ${
                    activeSessionId === s.id && currentTab === 'research'
                      ? 'bg-[#171a21] text-white font-medium border border-[#353c4b]'
                      : 'text-[#94a3b8] hover:text-white hover:bg-[#171a21]/50'
                  }`}
                  onClick={() => {
                    onSelectSession(s.id);
                    onSelectTab('research');
                  }}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileText className="w-3 h-3 shrink-0 text-[#64748b]" />
                    <span className="truncate">{s.title || 'Untitled Inquiry'}</span>
                  </div>

                  {onDeleteSession && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(s.id);
                      }}
                      className="opacity-0 group-hover/item:opacity-100 p-1 text-[#64748b] hover:text-rose-400 transition"
                      title="Delete session"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {isCollapsed && <div className="flex-1" />}

      {/* Bottom Footer & Settings Button */}
      <div className="p-3 border-t border-[#232732] bg-[#090a0d]/70">
        <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
          {!isCollapsed && (
            <div className="flex items-center gap-2 truncate">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="text-[11px] text-[#94a3b8] truncate capitalize font-mono">
                {providerName}
              </span>
            </div>
          )}

          <button
            onClick={onOpenSettings}
            className="p-1.5 rounded-lg text-[#94a3b8] hover:text-white hover:bg-[#171a21] transition cursor-pointer"
            title="Configure Engine Settings"
          >
            <SettingsIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
};
