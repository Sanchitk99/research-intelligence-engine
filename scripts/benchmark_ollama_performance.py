"""Controlled performance benchmark for Ollama gemma3:4b and Mock provider.

Measures:
1. Mock provider baseline (isolates retrieval, prompt prep, and citation validation from LLM)
2. Cold Ollama request (model unloaded via keep_alive: 0)
3. Multiple warm Ollama requests (median, p95, min, max, throughput)
4. Stage-level timing breakdown for each stage
"""

import json
import time
import statistics
import requests

API_URL = "http://127.0.0.1:8000/api/research/ask"
OLLAMA_BASE = "http://127.0.0.1:11434"
QUESTION = "What is Dense Passage Retrieval (DPR) and how does it compare to BM25?"

def run_query(provider: str, model: str = None, top_k: int = 5, retrieval_mode: str = "hybrid") -> dict:
    payload = {
        "question": QUESTION,
        "provider": provider,
        "model": model,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
    }
    t0 = time.perf_counter()
    res = requests.post(API_URL, json=payload, timeout=120)
    client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    res.raise_for_status()
    data = res.json()
    data["client_observed_ms"] = round(client_elapsed_ms, 2)
    return data

def unload_ollama_model(model_name: str = "gemma3:4b"):
    """Force Ollama to unload model from VRAM."""
    try:
        requests.post(f"{OLLAMA_BASE}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=10)
        time.sleep(2)
    except Exception as e:
        print(f"Warning unloading model: {e}")

def main():
    print("=" * 70)
    print("RESEARCH INTELLIGENCE ENGINE — PERFORMANCE BENCHMARK")
    print(f"Target URL: {API_URL}")
    print(f"Benchmark Question: '{QUESTION}'")
    print("=" * 70)

    # 1. Mock Provider Baseline
    print("\n--- 1. Mock Provider Baseline (Retrieval & Application Overhead) ---")
    mock_res = run_query(provider="mock", model="mock-model")
    print(f"Mock Answer Status: {mock_res.get('status')}")
    print(f"Reported Latency: {mock_res.get('latency_ms')} ms")
    print(f"Client Observed Latency: {mock_res.get('client_observed_ms')} ms")
    print(f"Timing Breakdown: {json.dumps(mock_res.get('timing_breakdown'), indent=2)}")

    # 2. Cold Ollama Request
    print("\n--- 2. Cold Ollama Request (Forced Unload before query) ---")
    print("Unloading gemma3:4b from Ollama...")
    unload_ollama_model("gemma3:4b")
    cold_res = run_query(provider="ollama", model="gemma3:4b")
    print(f"Cold Answer Status: {cold_res.get('status')}")
    print(f"Prompt Tokens: {cold_res.get('prompt_tokens')}, Completion Tokens: {cold_res.get('completion_tokens')}")
    print(f"Reported Latency: {cold_res.get('latency_ms')} ms")
    print(f"Client Observed Latency: {cold_res.get('client_observed_ms')} ms")
    print(f"Timing Breakdown: {json.dumps(cold_res.get('timing_breakdown'), indent=2)}")

    # 3. Repeated Warm Ollama Requests
    print("\n--- 3. Repeated Warm Ollama Requests (5 runs) ---")
    warm_latencies = []
    warm_llm_times = []
    warm_tps = []

    for i in range(1, 6):
        print(f"\n[Run {i}/5]")
        w_res = run_query(provider="ollama", model="gemma3:4b")
        lat = w_res.get("latency_ms")
        bd = w_res.get("timing_breakdown") or {}
        llm_t = bd.get("llm_generation_ms", 0.0)
        tps = bd.get("tokens_per_second", 0.0)
        warm_latencies.append(lat)
        warm_llm_times.append(llm_t)
        if tps:
            warm_tps.append(tps)
        print(f"  Total Latency: {lat} ms | LLM Gen: {llm_t} ms | Throughput: {tps} tokens/s")
        print(f"  Retrieval: {bd.get('retrieval_ms')} ms (Dense: {bd.get('dense_retrieval_ms')}, BM25: {bd.get('bm25_retrieval_ms')})")
        print(f"  Citation Validation: {bd.get('citation_validation_ms')} ms")

    # 4. Statistical Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK SUMMARY (WARM RUNS)")
    print("=" * 70)
    print(f"Total Warm Latency (ms):")
    print(f"  Min:    {min(warm_latencies):.2f} ms ({min(warm_latencies)/1000:.2f} s)")
    print(f"  Median: {statistics.median(warm_latencies):.2f} ms ({statistics.median(warm_latencies)/1000:.2f} s)")
    print(f"  Mean:   {statistics.mean(warm_latencies):.2f} ms ({statistics.mean(warm_latencies)/1000:.2f} s)")
    print(f"  Max:    {max(warm_latencies):.2f} ms ({max(warm_latencies)/1000:.2f} s)")
    if len(warm_latencies) > 1:
        print(f"  Stdev:  {statistics.stdev(warm_latencies):.2f} ms")
    if warm_tps:
        print(f"Inference Throughput (tokens/s):")
        print(f"  Median: {statistics.median(warm_tps):.2f} tokens/s")
        print(f"  Max:    {max(warm_tps):.2f} tokens/s")
    print("=" * 70)

if __name__ == "__main__":
    main()

