# Research Intelligence Engine (RIE) — Project Documentation

Welcome to the comprehensive technical documentation and system audit for the **Research Intelligence Engine (RIE)**.

This documentation suite provides a deep, production-grade examination of every aspect of the engine: its design philosophy, architectural components, end-to-end execution pipelines, algorithms, database models, frontend interfaces, security posture, empirical evaluation benchmarks, and code audit.

---

## 📚 Documentation Suite Navigation

The documentation is organized into 12 modular, highly detailed technical guides:

| Document | Primary Focus | Target Audience |
|---|---|---|
| [**00_PROJECT_OVERVIEW.md**](./00_PROJECT_OVERVIEW.md) | High-level mission, core problem statement, architectural philosophy, complete technology stack, and implementation status scorecard. | System Architects, Hiring Managers, Technical Evaluators |
| [**01_FOLDER_AND_FILE_GUIDE.md**](./01_FOLDER_AND_FILE_GUIDE.md) | Exhaustive directory tree and file-by-file responsibility catalog covering all 80+ source files in the repository. | New Contributors, Code Reviewers, Maintainers |
| [**02_CODEBASE_DEEP_DIVE.md**](./02_CODEBASE_DEEP_DIVE.md) | Detailed file-by-file code analysis: class signatures, core functions, inputs/outputs, error handling, validation, and design decisions. | Senior Software Engineers, Code Auditors |
| [**03_ARCHITECTURE_AND_FLOWS.md**](./03_ARCHITECTURE_AND_FLOWS.md) | Visual system diagrams (Mermaid) and numbered, end-to-end execution traces for question answering, ingestion, routing, and abstention. | System Architects, Integration Engineers |
| [**04_FRONTEND_AND_API.md**](./04_FRONTEND_AND_API.md) | Next.js 15 UI architecture, component hierarchy, state management, user journeys, and RESTful FastAPI endpoint reference. | Full-Stack Developers, Frontend Engineers |
| [**05_DATABASE_AND_DATA.md**](./05_DATABASE_AND_DATA.md) | Dual-storage data architecture: local embedded Qdrant vector database (RocksDB) and Okapi BM25 JSON corpus; ER diagrams and data lifecycles. | Database Engineers, Data Architects |
| [**06_ALGORITHMS_AND_ML.md**](./06_ALGORITHMS_AND_ML.md) | Mathematical and algorithmic formulations: boundary-aware chunking, dense embeddings, BM25, Reciprocal Rank Fusion ($k=60$), intent routing, and empirical evaluation metrics. | ML Engineers, Information Retrieval Specialists |
| [**07_SECURITY_AND_CONFIGURATION.md**](./07_SECURITY_AND_CONFIGURATION.md) | Security threat model, prompt injection defense, citation provenance enforcement, credential redaction, and environment variable reference. | Security Auditors, DevOps / DevSecOps |
| [**08_SETUP_TESTING_AND_DEPLOYMENT.md**](./08_SETUP_TESTING_AND_DEPLOYMENT.md) | Verified step-by-step setup guides, dependency installation, running FastAPI and Next.js, running pytest suites, and containerization notes. | DevOps, QA Engineers, Developers |
| [**09_AUDIT_AND_IMPROVEMENTS.md**](./09_AUDIT_AND_IMPROVEMENTS.md) | Code quality audit, identified risks, technical debt, testing gaps, and a prioritized engineering improvement roadmap. | Engineering Leads, Product Managers |
| [**10_INTERVIEW_PREPARATION.md**](./10_INTERVIEW_PREPARATION.md) | Technical interview preparation: 2-minute architectural pitch, deep-dive technical questions, tough edge cases, trade-off defenses, and whiteboarding guide. | Candidates, Engineering Interviewees |
| [**11_FILE_COVERAGE_INDEX.md**](./11_FILE_COVERAGE_INDEX.md) | Complete audit inventory: every file inspected, exclusions with justification, and audit verification checklist. | Compliance, Quality Assurance |

---

## 🎯 Recommended Reading Pathways

Depending on your role and objectives, we recommend the following reading sequences:

### Pathway A: System Architect / Engineering Lead (High-Level Deep Dive)
1. [`00_PROJECT_OVERVIEW.md`](./00_PROJECT_OVERVIEW.md) — Understand design goals, architecture, and current state.
2. [`03_ARCHITECTURE_AND_FLOWS.md`](./03_ARCHITECTURE_AND_FLOWS.md) — Inspect data flows, component interactions, and failure paths.
3. [`06_ALGORITHMS_AND_ML.md`](./06_ALGORITHMS_AND_ML.md) — Review retrieval mathematics, RRF fusion, and empirical benchmarks.
4. [`09_AUDIT_AND_IMPROVEMENTS.md`](./09_AUDIT_AND_IMPROVEMENTS.md) — Evaluate technical debt and roadmap priorities.

### Pathway B: Full-Stack Developer / Contributor (Hands-On Implementation)
1. [`01_FOLDER_AND_FILE_GUIDE.md`](./01_FOLDER_AND_FILE_GUIDE.md) — Familiarize yourself with the workspace layout.
2. [`08_SETUP_TESTING_AND_DEPLOYMENT.md`](./08_SETUP_TESTING_AND_DEPLOYMENT.md) — Spin up the FastAPI backend and Next.js frontend locally.
3. [`04_FRONTEND_AND_API.md`](./04_FRONTEND_AND_API.md) — Review API contracts and React UI components.
4. [`02_CODEBASE_DEEP_DIVE.md`](./02_CODEBASE_DEEP_DIVE.md) — Study the implementation of modules you intend to touch.

### Pathway C: Technical Recruiter / Peer Reviewer / Interviewee (Evaluation & Defense)
1. [`00_PROJECT_OVERVIEW.md`](./00_PROJECT_OVERVIEW.md) — Executive summary of capabilities and empirical validation.
2. [`10_INTERVIEW_PREPARATION.md`](./10_INTERVIEW_PREPARATION.md) — Master the 2-minute elevator pitch, technical trade-offs, and defense questions.
3. [`06_ALGORITHMS_AND_ML.md`](./06_ALGORITHMS_AND_ML.md) — Verify retrieval evaluation results (Hit@5 83.3% vs. 66.7%).
4. [`07_SECURITY_AND_CONFIGURATION.md`](./07_SECURITY_AND_CONFIGURATION.md) — Review citation provenance and anti-hallucination guarantees.

---

## 🔍 Evidence and Grounding Standard

Every claim, diagram, metric, and code reference in this documentation suite is strictly grounded in the committed codebase of this repository:
- **Retrieval Metrics** are derived from committed benchmark runs (`evaluation/results/retrieval_evaluation_report.md` and `evaluation/results/hybrid_retrieval_evaluation_report.md`).
- **Answer Quality Metrics** reflect the controlled 28-question evaluation suite (`evaluation/results/answer_quality_latency_report.md`).
- **File references** cite active repository paths, class names, and functions.
- Inferred design rationales are explicitly identified as engineering analysis.

