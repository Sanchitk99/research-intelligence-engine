'use client';

import React from 'react';
import {
  FileText,
  Search,
  Shuffle,
  ShieldCheck,
  ExternalLink,
  Layers,
  Database,
  Code2,
  Terminal,
  Cpu,
  GitBranch,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
} from 'lucide-react';

interface ProjectOverviewProps {
  totalPoints?: number;
}

export const ProjectOverview: React.FC<ProjectOverviewProps> = ({
  totalPoints = 396,
}) => {
  const GITHUB_REPO_URL = 'https://github.com/Sanchitk99/research-intelligence-engine';
  const HYBRID_REPORT_URL = `${GITHUB_REPO_URL}/blob/main/evaluation/results/hybrid_vs_dense_comparison_report.md`;
  const ROUTING_REPORT_URL = `${GITHUB_REPO_URL}/blob/main/evaluation/results/query_intent_routing_report.md`;
  const DENSE_REPORT_URL = `${GITHUB_REPO_URL}/blob/main/evaluation/results/retrieval_evaluation_report.md`;

  const stages = [
    {
      num: '1',
      title: 'Document Ingestion',
      subtitle: 'PyMuPDF & Boundary Chunker',
      desc: 'Page-by-page extraction tracking 1-based physical pages and metadata. Structural chunking preserves sentence and paragraph boundaries to prevent fragmented context.',
      icon: FileText,
      badge: '396 Chunks',
      color: 'text-indigo-400',
      bg: 'bg-indigo-950/20 border-indigo-800/30',
    },
    {
      num: '2',
      title: 'Hybrid Retrieval',
      subtitle: 'Dense SBERT + Sparse BM25',
      desc: 'Dense semantic vector search (SentenceTransformers all-MiniLM-L6-v2, 384-d cosine similarity in Qdrant) alongside sparse Okapi BM25 keyword matching for exact scientific terminology.',
      icon: Search,
      badge: 'Dual Retrieval',
      color: 'text-sky-400',
      bg: 'bg-sky-950/20 border-sky-800/30',
    },
    {
      num: '3',
      title: 'Rank Fusion',
      subtitle: 'Reciprocal Rank Fusion (k=60)',
      desc: 'RRF merges dense semantic and sparse lexical rankings without arbitrary score scaling. Optional query-intent router detects comparative queries to diversify cross-paper evidence.',
      icon: Shuffle,
      badge: 'RRF k=60',
      color: 'text-violet-400',
      bg: 'bg-violet-950/20 border-violet-800/30',
    },
    {
      num: '4',
      title: 'Evidence-Grounded Answering',
      subtitle: 'Schema Enforcement & Abstention',
      desc: 'Structured synthesis with automated citation validation against retrieved context. Ungrounded citations are rejected; questions lacking evidence trigger explicit abstention with 0 citations.',
      icon: ShieldCheck,
      badge: 'Zero Hallucinations',
      color: 'text-emerald-400',
      bg: 'bg-emerald-950/20 border-emerald-800/30',
    },
  ];

  const techStack = [
    {
      category: 'Backend & API',
      icon: Terminal,
      items: [
        { name: 'Python 3.10+', role: 'Core runtime' },
        { name: 'FastAPI', role: 'RESTful API service' },
        { name: 'Pydantic v2', role: 'Strict schema contracts' },
        { name: 'PyMuPDF', role: 'PDF parsing & pagination' },
      ],
    },
    {
      category: 'Frontend & UI',
      icon: Code2,
      items: [
        { name: 'Next.js 15', role: 'React application framework' },
        { name: 'React 19', role: 'Component interface' },
        { name: 'TypeScript', role: 'End-to-end type safety' },
        { name: 'Tailwind CSS', role: 'Dark editorial aesthetics' },
      ],
    },
    {
      category: 'Retrieval & Storage',
      icon: Database,
      items: [
        { name: 'Qdrant (Embedded)', role: 'Dense vector database' },
        { name: 'SentenceTransformers', role: 'all-MiniLM-L6-v2 (384-d)' },
        { name: 'Rank-BM25', role: 'Okapi BM25 lexical index' },
        { name: 'RRF (k=60)', role: 'Reciprocal Rank Fusion' },
      ],
    },
    {
      category: 'Research Workflow',
      icon: Cpu,
      items: [
        { name: 'Sentence-Aware Chunking', role: 'Verbatim integrity' },
        { name: 'Grounded Synthesizer', role: 'Multi-LLM providers' },
        { name: 'Citation Validator', role: 'Page & chunk verification' },
        { name: 'Abstention Safeguards', role: 'Defensive 0-citation fallback' },
      ],
    },
  ];

  return (
    <div className="space-y-8 animate-fade-in max-w-5xl mx-auto pb-12">
      {/* Header Summary */}
      <div className="border-b border-[#232732] pb-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            About Research Intelligence Engine
          </h2>
          <p className="text-xs text-[#94a3b8] mt-1 max-w-2xl leading-relaxed">
            An evidence-grounded research intelligence prototype that synthesizes answers across
            scientific literature with verified, page-level citations and strict abstention safeguards.
          </p>
        </div>

        <a
          href={GITHUB_REPO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[#111318] hover:bg-[#171a21] text-xs font-medium text-white border border-[#232732] hover:border-[#3b4455] transition shadow-sm self-start md:self-auto cursor-pointer"
        >
          <GitBranch className="w-3.5 h-3.5 text-indigo-400" />
          <span>View on GitHub</span>
          <ExternalLink className="w-3 h-3 text-[#64748b]" />
        </a>
      </div>

      {/* SECTION A: HOW IT WORKS */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-500" />
              1. How It Works — 4-Stage Architecture Flow
            </h3>
            <p className="text-[11px] text-[#94a3b8] mt-0.5">
              Integrated modular pipeline from PDF ingestion to evidence-verified synthesis
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {stages.map((stage, idx) => {
            const Icon = stage.icon;
            return (
              <div
                key={stage.num}
                className={`relative p-4 rounded-xl border flex flex-col justify-between gap-3 ${stage.bg} transition hover:border-[#3b4455]`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2.5">
                    <span className="w-6 h-6 rounded-full bg-[#111318] border border-[#232732] text-[11px] font-mono font-bold text-white flex items-center justify-center">
                      {stage.num}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#111318]/80 text-[#94a3b8] border border-[#232732]">
                      {stage.badge}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 mb-1">
                    <Icon className={`w-4 h-4 ${stage.color} shrink-0`} />
                    <h4 className="text-xs font-semibold text-white">{stage.title}</h4>
                  </div>

                  <span className="text-[10px] font-mono font-medium text-indigo-300 block mb-2">
                    {stage.subtitle}
                  </span>

                  <p className="text-[11px] text-[#94a3b8] leading-relaxed">
                    {stage.desc}
                  </p>
                </div>

                {idx < stages.length - 1 && (
                  <div className="hidden md:block absolute -right-2 top-1/2 -translate-y-1/2 z-10">
                    <div className="w-4 h-4 rounded-full bg-[#111318] border border-[#232732] flex items-center justify-center text-[#64748b]">
                      <ArrowRight className="w-2.5 h-2.5" />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="p-3.5 rounded-lg bg-[#111318] border border-[#232732] text-[11px] text-[#94a3b8] flex items-start gap-2.5">
          <Layers className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="text-white font-medium">Modular In-Process Pipeline:</strong> Stages operate
            as an integrated pipeline within the FastAPI service rather than separate microservices.
            The core hybrid retrieval pipeline executes unconstrained by default, while
            lightweight <strong className="text-indigo-300 font-medium">Query-Intent Routing</strong> can be
            toggled to dynamically apply per-document caps only when multi-paper comparative queries are detected.
          </p>
        </div>
      </section>

      {/* SECTION B: EMPIRICAL RETRIEVAL BENCHMARKS */}
      <section className="space-y-4">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            2. Empirical Retrieval Performance
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Quantitative evaluation across 18 answerable benchmark questions from 5 seminal NLP research papers
          </p>
        </div>

        {/* Benchmark Metric Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748b]">
              Dense Baseline Hit@5
            </span>
            <div className="text-xl font-bold font-mono text-[#94a3b8]">66.7%</div>
            <p className="text-[10px] text-[#64748b]">Dense-only (all-MiniLM-L6-v2)</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111318] border border-emerald-800/40 bg-emerald-950/10 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400">
                Hybrid RRF Hit@5
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/60">
                +16.6%
              </span>
            </div>
            <div className="text-xl font-bold font-mono text-emerald-400">83.3%</div>
            <p className="text-[10px] text-[#94a3b8]">Dense + Okapi BM25 (RRF k=60)</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748b]">
              Hybrid RRF MRR@5
            </span>
            <div className="text-xl font-bold font-mono text-white">0.4509</div>
            <p className="text-[10px] text-[#64748b]">Mean Reciprocal Rank (vs. 0.4167)</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-1">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#64748b]">
              Evaluation Set
            </span>
            <div className="text-xl font-bold font-mono text-indigo-400">18 Questions</div>
            <p className="text-[10px] text-[#64748b]">Gold standard across P1–P5</p>
          </div>
        </div>

        {/* Important Qualification Alert */}
        <div className="p-4 rounded-xl bg-[#171a21] border border-[#353c4b] text-xs text-[#cbd5e1] space-y-2">
          <div className="flex items-center gap-2 text-indigo-300 font-semibold text-xs">
            <CheckCircle2 className="w-4 h-4 text-indigo-400 shrink-0" />
            <span>Benchmark Qualification & Scope Note</span>
          </div>
          <p className="leading-relaxed text-[#94a3b8]">
            &ldquo;These figures measure retrieval performance on the documented benchmark, not end-to-end
            answer accuracy or guaranteed citation correctness.&rdquo;
          </p>
          <p className="text-[11px] text-[#64748b] leading-relaxed border-t border-[#232732] pt-2">
            <strong className="text-[#94a3b8]">Query-Intent Routing Evaluation:</strong> Query-intent routing
            was evaluated separately on 37 queries (9 cross-paper pairs, 18 answerable, 10 out-of-scope).
            While blind document caps boost cross-paper coverage from 55.6% to 100%, they cause an unintended
            -27.8% recall drop on single-paper questions. Intent routing achieves 100% comparative coverage
            while shielding single-paper recall at 72.2%. It remains <strong className="text-white">opt-in</strong> in
            Settings to guarantee baseline invariance for standard inquiries.
          </p>
        </div>

        {/* Evaluation Report Links */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] text-[#64748b]">Verified Evaluation Reports:</span>
          <a
            href={HYBRID_REPORT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111318] hover:bg-[#171a21] border border-[#232732] text-[11px] font-mono text-indigo-300 hover:text-white transition cursor-pointer"
          >
            <FileSpreadsheet className="w-3 h-3 text-indigo-400" />
            <span>hybrid_vs_dense_comparison_report.md</span>
            <ExternalLink className="w-2.5 h-2.5 text-[#64748b]" />
          </a>

          <a
            href={ROUTING_REPORT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111318] hover:bg-[#171a21] border border-[#232732] text-[11px] font-mono text-indigo-300 hover:text-white transition cursor-pointer"
          >
            <FileSpreadsheet className="w-3 h-3 text-indigo-400" />
            <span>query_intent_routing_report.md</span>
            <ExternalLink className="w-2.5 h-2.5 text-[#64748b]" />
          </a>

          <a
            href={DENSE_REPORT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#111318] hover:bg-[#171a21] border border-[#232732] text-[11px] font-mono text-indigo-300 hover:text-white transition cursor-pointer"
          >
            <FileSpreadsheet className="w-3 h-3 text-indigo-400" />
            <span>retrieval_evaluation_report.md</span>
            <ExternalLink className="w-2.5 h-2.5 text-[#64748b]" />
          </a>
        </div>
      </section>

      {/* SECTION C: TECHNICAL OVERVIEW & CORPUS */}
      <section className="space-y-4">
        <div>
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-violet-500" />
            3. Technical Stack & Prototype Constraints
          </h3>
          <p className="text-[11px] text-[#94a3b8] mt-0.5">
            Architecture stack verified against repository dependencies and engineering design
          </p>
        </div>

        {/* Technology Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {techStack.map((group) => {
            const Icon = group.icon;
            return (
              <div
                key={group.category}
                className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-3"
              >
                <div className="flex items-center gap-2 pb-2 border-b border-[#232732]">
                  <Icon className="w-4 h-4 text-indigo-400" />
                  <h4 className="text-xs font-semibold text-white">{group.category}</h4>
                </div>

                <ul className="space-y-2">
                  {group.items.map((item) => (
                    <li key={item.name} className="flex flex-col">
                      <span className="text-xs font-medium text-[#cbd5e1]">{item.name}</span>
                      <span className="text-[10px] text-[#64748b]">{item.role}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>

        {/* Corpus Details & Constraints */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          {/* Corpus Statistics */}
          <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-2">
            <div className="flex items-center gap-2 text-white font-semibold text-xs">
              <Database className="w-4 h-4 text-emerald-400" />
              <span>Literature Corpus Scope</span>
            </div>
            <p className="text-[11px] text-[#94a3b8] leading-relaxed">
              Indexed collection comprises <strong className="text-white">5 seminal research papers</strong> (RAG,
              DPR, REALM, Self-RAG, Ragas) encompassing 82 pages and{' '}
              <strong className="text-emerald-400 font-mono">{totalPoints} dense vector chunks</strong> stored
              in the local collection <code className="text-[10px] font-mono text-indigo-300">research_corpus</code>.
            </p>
          </div>

          {/* Prototype Scope & Concurrency Warning */}
          <div className="p-4 rounded-xl bg-[#111318] border border-[#232732] space-y-2">
            <div className="flex items-center gap-2 text-amber-400 font-semibold text-xs">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              <span>Storage Concurrency & Prototype Bounds</span>
            </div>
            <p className="text-[11px] text-[#94a3b8] leading-relaxed">
              Local Qdrant operates in embedded filesystem mode with an exclusive lock (<code className="text-[10px] font-mono text-amber-300">portalocker</code>).
              Direct offline Python scripts require stopping the background FastAPI daemon to acquire the lock.
              Session history persists locally in browser <code className="text-[10px] font-mono text-indigo-300">localStorage</code>.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
