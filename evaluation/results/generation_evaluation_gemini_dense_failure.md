# Live Gemini Generation Evaluation — Failed

- **Mode:** `dense`
- **Provider:** `gemini`
- **Model:** `gemini-flash-lite-latest`
- **Failure type:** `GenerationEvaluationError`
- **Failure stage:** `generation`
- **Question:** `Q16` (answerable)
- **Questions:** 16 attempted, 15 completed, 1 failed
- **Provider calls:** 17 opportunities attempted, 16 completed, 1 failed
- **Failure:** Generation failed for Q16: Gemini rate limit exceeded: {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, model: gemini-3.5-flash-lite\nPlease retry in 25.368622874s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.Help",
        "links": [
          {
            "description": "Learn more about Gemini API quotas",
            "url": "https://ai.google.dev/gemini-api/docs/rate-limits"
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
          {
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            "quotaDimensions": {
              "model": "gemini-3.5-flash-lite",
              "location": "global"
            },
            "quotaValue": "15"
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "25s"
      }
    ]
  }
}


The live evaluation stopped without treating the partial run as a successful benchmark.
