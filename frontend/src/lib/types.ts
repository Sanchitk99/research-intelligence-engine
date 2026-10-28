/**
 * TypeScript data models for the Research Intelligence Engine frontend.
 * Strictly aligned with FastAPI backend Pydantic schemas.
 */

export type AnswerStatusType =
  | 'answered'
  | 'partially_answered'
  | 'abstained_insufficient_evidence'
  | 'abstained_out_of_scope'
  | 'conflicting_evidence'
  | 'error';

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  physical_page_number: number;
  printed_page_number?: number | null;
  excerpt: string;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  document_title: string;
  physical_page_number: number;
  printed_page_number?: number | null;
  text: string;
  score: number;
}

export interface AskQuestionRequest {
  question: string;
  provider?: string;
  model?: string | null;
  retrieval_mode?: 'hybrid' | 'dense';
  top_k?: number;
  min_score?: number | null;
  enable_intent_routing?: boolean;
}

export interface AskQuestionResponse {
  question: string;
  answer: string;
  status: AnswerStatusType | string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  model_name: string;
  provider_name: string;
  retrieval_mode: string;
  latency_ms: number;
  confidence_notes?: string | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
}

export interface PaperSummary {
  document_id: string;
  title: string;
  total_pages: number;
  total_chunks: number;
}

export interface CorpusPDF {
  filename: string;
  document_id: string;
  title: string;
  file_size_bytes: number;
}

export interface IngestionResponse {
  success: boolean;
  document_id: string;
  title: string;
  filename: string;
  pages_extracted: number;
  chunks_created: number;
  chunks_indexed: number;
  total_collection_points: number;
  error_message?: string | null;
  sample_chunks: Array<{ chunk_id?: string; preview?: string; [key: string]: unknown }>;
  dry_run?: boolean;
}

export interface ProviderStatus {
  configured: boolean;
  default_model: string;
}

export interface SystemStatus {
  collection_name: string;
  total_points: number;
  indexed_papers_count: number;
  papers: PaperSummary[];
  providers: Record<string, ProviderStatus>;
}

export interface QAPair {
  id: string;
  question: string;
  timestamp: string;
  response: AskQuestionResponse;
}

export interface ResearchSession {
  id: string;
  title: string;
  createdAt: string;
  history: QAPair[];
}

export interface EngineSettings {
  provider: string;
  model: string;
  retrieval_mode: 'hybrid' | 'dense';
  top_k: number;
  enable_intent_routing: boolean;
}
