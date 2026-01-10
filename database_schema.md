┌─────────────────────────────────────────────────────────────┐
│                     request_logs                            │
├─────────────────┬──────────────┬────────────────────────────┤
│ Column          │ Type         │ Description                │
├─────────────────┼──────────────┼────────────────────────────┤
│ request_id      │ TEXT (PK)    │ UUID for the request       │
│ timestamp       │ DATETIME     │ When request was received  │
│ user_id         │ INTEGER      │ Requested user ID          │
│ endpoint        │ TEXT         │ /recommend, /similar, etc  │
│ model_version   │ TEXT         │ Model version used         │
│ recommendations │ TEXT (JSON)  │ List of movie IDs          │
│ scores          │ TEXT (JSON)  │ List of scores             │
│ latency_ms      │ REAL         │ Response time              │
│ is_fallback     │ INTEGER      │ 1 if fallback used         │
│ fallback_reason │ TEXT         │ Why fallback was used      │
└─────────────────┴──────────────┴────────────────────────────┘
