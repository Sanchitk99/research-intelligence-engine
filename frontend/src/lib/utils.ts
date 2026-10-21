import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number, decimals = 1): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function getStatusBadgeConfig(status: string): { label: string; bg: string; text: string; border: string; dot: string } {
  switch (status.toLowerCase()) {
    case 'answered':
      return {
        label: 'Grounded in Evidence',
        bg: 'bg-emerald-950/40',
        text: 'text-emerald-300',
        border: 'border-emerald-700/50',
        dot: 'bg-emerald-400',
      };
    case 'partially_answered':
      return {
        label: 'Partially Answered',
        bg: 'bg-amber-950/40',
        text: 'text-amber-300',
        border: 'border-amber-700/50',
        dot: 'bg-amber-400',
      };
    case 'abstained_out_of_scope':
      return {
        label: 'Abstained: Out of Scope',
        bg: 'bg-indigo-950/40',
        text: 'text-indigo-300',
        border: 'border-indigo-700/50',
        dot: 'bg-indigo-400',
      };
    case 'abstained_insufficient_evidence':
      return {
        label: 'Abstained: Insufficient Evidence',
        bg: 'bg-slate-900',
        text: 'text-slate-300',
        border: 'border-slate-700/60',
        dot: 'bg-slate-400',
      };
    case 'conflicting_evidence':
      return {
        label: 'Conflicting Evidence',
        bg: 'bg-purple-950/40',
        text: 'text-purple-300',
        border: 'border-purple-700/50',
        dot: 'bg-purple-400',
      };
    case 'error':
    default:
      return {
        label: 'Generation Error',
        bg: 'bg-rose-950/40',
        text: 'text-rose-300',
        border: 'border-rose-700/50',
        dot: 'bg-rose-400',
      };
  }
}
