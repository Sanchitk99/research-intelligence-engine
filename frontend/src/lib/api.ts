import {
  AskQuestionRequest,
  AskQuestionResponse,
  SystemStatus,
  PaperSummary,
  CorpusPDF,
  IngestionResponse,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!res.ok) {
      let errorDetail = `Request failed with status ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson?.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fall back to status text
      }
      throw new Error(errorDetail);
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error('Network error: Unable to connect to Research Intelligence Engine API.');
  }
}

export const api = {
  // System Health & Status
  async getHealth(): Promise<{ status: string; service: string; version: string }> {
    return request('/api/system/health');
  },

  async getSystemStatus(): Promise<SystemStatus> {
    return request('/api/system/status');
  },

  // Research Q&A
  async askQuestion(payload: AskQuestionRequest): Promise<AskQuestionResponse> {
    return request('/api/research/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // Papers & Corpus
  async getIndexedPapers(): Promise<PaperSummary[]> {
    return request('/api/papers');
  },

  async getCorpusPDFs(): Promise<CorpusPDF[]> {
    return request('/api/papers/corpus');
  },

  async ingestCorpusPaper(
    filename: string,
    dryRun = false,
    docId?: string,
    docTitle?: string
  ): Promise<IngestionResponse> {
    return request('/api/papers/ingest', {
      method: 'POST',
      body: JSON.stringify({
        filename,
        dry_run: dryRun,
        doc_id: docId || null,
        doc_title: docTitle || null,
      }),
    });
  },

  async uploadAndIngestPDF(
    file: File,
    dryRun = false,
    docId?: string,
    docTitle?: string
  ): Promise<IngestionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dry_run', String(dryRun));
    if (docId) formData.append('doc_id', docId);
    if (docTitle) formData.append('doc_title', docTitle);

    const url = `${API_BASE_URL}/api/papers/upload`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      // Note: do not set Content-Type header so browser sets multipart boundary automatically
    });

    if (!res.ok) {
      let detail = `Upload failed with status ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson?.detail) detail = errorJson.detail;
      } catch {
        // Fall back to detail
      }
      throw new Error(detail);
    }

    return await res.json();
  },
};
