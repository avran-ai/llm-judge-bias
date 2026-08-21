verdicts 768 over 48 prompts · recognition probes 378
per judge: {'GPT-5.6 SOL': 192, 'Claude Opus 4.8': 192, 'Gemini 3.5 Flash': 192, 'DeepSeek V4 Flash': 192}

TABLE 1 — SELF-PREFERENCE
| Judge | Votes own family | Other judges vote that family | Excess |
| --- | --- | --- | --- |
| GPT-5.6 SOL | 85% | 38% | **+0.47** [+0.39, +0.55] |
| Claude Opus 4.8 | 27% | 11% | **+0.16** [+0.09, +0.25] |
| Gemini 3.5 Flash | 24% | 12% | **+0.13** [+0.05, +0.22] |
| DeepSeek V4 Flash | 25% | 19% | +0.06 [-0.01, +0.13]  (crosses zero) |

TABLE 2 — SELF-RECOGNITION (chance = 25%)
| Judge | Correct | Rate | Order 0 | Order 1 |
| --- | --- | --- | --- | --- |
| GPT-5.6 SOL | 81/96 | **84%** | 83% | 85% |
| Gemini 3.5 Flash | 49/96 | **51%** | 54% | 48% |
| Claude Opus 4.8 | 39/96 | **41%** | 42% | 40% |
| DeepSeek V4 Flash | 15/90 | **17%** | 17% | 16% |

TABLE 3 — ORDERING BIAS (unbiased = 25% per slot)
| Judge | Slot 1 | Slot 2 | Slot 3 | Slot 4 | chi2 | first half vs second |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.6 SOL | 26% | 28% | 24% | 22% | 1.2 (ns) | 103 vs 89, p=3.5e-01 |
| Claude Opus 4.8 | 22% | 19% | 29% | 29% | 5.7 (ns) | 80 vs 112, p=2.5e-02 |
| Gemini 3.5 Flash | 35% | 20% | 23% | 22% | 10.5 (significant) | 105 vs 87, p=2.2e-01 |
| DeepSeek V4 Flash | 30% | 32% | 19% | 20% | 10.3 (significant) | 118 vs 74, p=1.8e-03 |

Slot occupancy check (each answer should sit in each slot equally):
         GPT-5.6 SOL: slot1=192 slot2=200 slot3=192 slot4=184
     Claude Opus 4.8: slot1=192 slot2=184 slot3=192 slot4=200
    Gemini 3.5 Flash: slot1=192 slot2=200 slot3=192 slot4=184
   DeepSeek V4 Flash: slot1=192 slot2=184 slot3=192 slot4=200

Verdict stability across the four orderings of identical answers:
         GPT-5.6 SOL: same winner all four times 40/48 · 1.23 distinct winners on average
     Claude Opus 4.8: same winner all four times 14/48 · 1.90 distinct winners on average
    Gemini 3.5 Flash: same winner all four times 18/48 · 1.79 distinct winners on average
   DeepSeek V4 Flash: same winner all four times 15/48 · 1.88 distinct winners on average

Does bias track recognition? (a prompt counts as recognised only if the judge identified itself in both orderings)
| Judge | Recognition | Overall excess | Excess when recognised | When missed |
| --- | --- | --- | --- | --- |
| GPT-5.6 SOL | 84% | +0.47 | +0.50 (n=39) | +0.08 (n=6) |
| Claude Opus 4.8 | 41% | +0.16 | +0.44 (n=13) | +0.03 (n=22) |
| Gemini 3.5 Flash | 51% | +0.13 | +0.07 (n=18) | +0.23 (n=17) |
| DeepSeek V4 Flash | 17% | +0.06 | +0.31 (n=3) | +0.02 (n=34) |

Total API spend recorded in the data files: $8.74
