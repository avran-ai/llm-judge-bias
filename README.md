# llm-judge-bias

Do LLM judges favour their own answers, and can they tell which answers are
theirs?

Four models from four families answered the same prompts, then judged each
other blind, then were asked to pick their own answer out of the line-up.
Everything here is the actual code and the actual data behind the write-up.

Reproducing it costs roughly **$9** and about an hour, most of it spent
waiting on the slower models.

## What was found

**One judge favours its own answers enormously, two do so mildly, one not at
all.** The excess is the judge's own-family vote rate minus the rate the
*other* judges give the same answers, so a family that genuinely wrote better
answers does not count as bias. Intervals are 95% bootstrap CIs resampled over
prompts.

Read the Method notes before quoting the two middle rows: this peer baseline
is contaminated by SOL's self-voting, and against a neutral judge Claude and
Gemini roughly halve.

| Judge | Votes own family | Other judges vote that family | Excess |
| --- | --- | --- | --- |
| GPT-5.6 SOL | 85% | 38% | **+0.47** [+0.39, +0.55] |
| Claude Opus 4.8 | 27% | 11% | **+0.16** [+0.09, +0.25] |
| Gemini 3.5 Flash | 24% | 12% | **+0.13** [+0.05, +0.22] |
| DeepSeek V4 Flash | 25% | 19% | +0.06 [-0.01, +0.13] (crosses zero) |

**Three of the four can pick their own answer out of four,** well above the
25% you would get by guessing, and the score barely moves when the answers are
shown in a different order.

| Judge | Correct | Rate |
| --- | --- | --- |
| GPT-5.6 SOL | 81/96 | 84% |
| Gemini 3.5 Flash | 49/96 | 51% |
| Claude Opus 4.8 | 39/96 | 41% |
| DeepSeek V4 Flash | 15/90 | 17% |

**The two results line up, though the split is thin.** For SOL and Claude the
bias concentrates on the prompts where the judge identified its own answer:
SOL +0.50 (n=39) against +0.08 (n=6) when it failed to, Claude +0.44 (n=13)
against +0.03 (n=22). Note SOL's miss set is only six prompts and its
clustered CI there is [-0.06, +0.22], so that half of the contrast is
directional at best. Two of those six are known-bad items (`p0`, `p38`) where
recognition is impossible by construction. DeepSeek's own split is reported at
n=1 and should be ignored entirely.

DeepSeek is better read as a natural control: it is the only judge that cannot
find its own writing, scoring below chance, and the only judge with no
measurable self-preference. Gemini is the exception, its split runs the other
way, so its smaller bias is probably not recognition-driven.

**Ordering matters more than expected.** DeepSeek clearly favours answers
shown early (first half 118 against 74, p=0.0018). Gemini's effect is
narrower: it over-picks slot 1 specifically, but its first-half split is 105
against 87, which is not significant. Claude leans the other way and does
significantly favour late slots (80 against 112, p=0.025) even though its
four-slot chi-squared does not reach significance. So three of the four judges
show some position effect, in two different directions.

Worse, show the same judge the same four answers in four different orders and
the winner often changes: Gemini gave the same answer all four times on 18 of
48 prompts, DeepSeek 15, Claude 14. SOL manages 40 of 48 only because it votes
for itself wherever its answer sits.

Full numbers, including the significance tests, are in
[results/RESULTS.md](results/RESULTS.md), and `analysis.py` regenerates all of
it from the raw data at no cost.

## Running it

```sh
cp .env.example .env                 # then put your key in it
set -a; source .env; set +a          # .env has no `export`, so -a is needed

python judge_bias.py gen      # 4 candidates answer 50 prompts
python judge_bias.py judge    # 4 judges x 48 prompts x 4 answer orderings
python judge_bias.py recog    # the "which one did you write" probe, 2 orderings
python judge_bias.py explain  # asks a few judges to justify their guess
python analysis.py            # the tables above, no network, no cost
```

Every phase is checkpointed to `data/*.jsonl` and is resumable: rerun the same
command and it only issues the calls still missing. A failed call is printed to stdout and
skipped rather than killing the run, which matters because one provider in this
set returns empty responses fairly often. Spending is capped per process with
`--max-spend` (default $15).

Model calls go through [OpenRouter](https://openrouter.ai), so a single key
covers all four families.

## Method notes

- **Candidates and judges are the same four models**, so every judge has one of
  its own answers in every line-up. There is no neutral outsider *on the panel*;
  each judge's comparison group is the other three.

  **This baseline is contaminated, and not in the safe direction.** An earlier
  draft of this README claimed the peer baseline made each excess
  "conservative". That is backwards. SOL takes 163 of its 192 votes for itself,
  so it almost never votes for anyone else's answers, which starves the peer
  baseline for the other three judges and *inflates* their excess. Measured
  against a genuinely neutral judge instead, the numbers shrink:

  | Judge | vs 3 peers (headline) | vs peers excluding SOL | vs neutral outsider |
  | --- | --- | --- | --- |
  | GPT-5.6 SOL | +0.467 | +0.467 | +0.427 |
  | Claude Opus 4.8 | +0.165 | +0.135 | +0.083 |
  | Gemini 3.5 Flash | +0.128 | +0.094 | +0.089 |
  | DeepSeek V4 Flash | +0.059 | -0.008 | +0.016 |

  SOL's effect is robust on every baseline. Claude and Gemini roughly halve
  against the outsider but stay positive. DeepSeek's already-insignificant
  effect disappears. `analysis.py` prints this table on every run.

  The neutral judge is **qwen/qwen3.7-max**, whose 192 verdicts on these exact
  answers ship in `data/verdicts.jsonl`. It was used in an earlier version of
  the experiment and dropped from the panel; the headline table keeps the
  four-way panel for comparability, but the outsider column above is the
  cleaner measurement and you should weight it accordingly.
- **Answers are anonymised and order-rotated.** Nothing in a judging prompt
  names a model, a provider, or who else is judging.
- **Temperature 0 throughout**, which is likely the worst case for
  self-recognition: a model's own answer is the most self-like text it can
  produce. Note that temperature 0 plus a fixed seed is best effort, not a
  guarantee, since providers are not perfectly deterministic. The cached data
  files, not the API, are the reproducible object here.
- **Bootstrap resamples whole prompts**, because four verdicts on one prompt
  are not independent observations. The chi-squared and binomial tests in the
  ordering section do NOT apply that correction and treat 192 correlated
  verdicts as independent. Re-running both prompt-clustered leaves every
  conclusion standing, so no reported number changes, but the p-values are
  more confident than they have strictly earned.
- **The rotation scheme is not a balanced Latin square.** Within one
  prompt-judge cell, two of the four answers do not visit all four slots.
  Aggregate occupancy is close (184 to 200 out of 768) and standardising by
  slot moves every excess by at most 0.006, but a cleaner rerun should use
  cyclic shifts 0 through 3 rather than three shifts plus a reversal. See
  `ordering()` in `judge_bias.py`.

## The prompts

Forty-eight prompts: 13 everyday questions, 13 writing tasks, 10 reasoning
problems, and 12 verifiable maths and code questions whose answer keys were
computed locally rather than written by hand. The `source` field records where
each came from: 18 are derived from WildChat, 18 from Alpaca, and 12 are
synthetic.

The original set was 50. DeepSeek returned empty content on two reasoning
prompts, so they never got a complete set of four answers and were never
judged. Both they and their six orphaned answers have been removed, so what
ships is exactly what was measured: 48 prompts, 192 answers, 768 verdicts.
Prompt ids keep their original numbering and therefore skip 28 and 35, because
the verdict and recognition files reference prompts by id and renumbering
would silently rewire the dataset.

**Known-bad items, left in for transparency:** `p0` produced four identical
answers so nothing could be judged, `p38` produced two identical ones, `p25`
contains an "ignore previous instructions" string, and one answer to `p26`
mentions a rival model by name. Excluding all four moves every excess figure
by at most 0.024, so the headline results do not depend on them.

## Rerunning it

`gen` and `judge` are complete: run them and they will issue zero calls. Two
phases are not, and both would change published numbers if you top them up:

- `recog` will retry 6 DeepSeek probes that its provider failed with empty
  responses. DeepSeek's recognition rate is 15/90; if those land it becomes a
  fraction of 96 and the reported 17% shifts.
- `explain` is a qualitative sampler rather than a measured phase, and ships
  partial coverage (13 rows). It will top up to 24 by default. Nothing in the
  three result tables depends on it.

## Repository layout

```
judge_bias.py      the harness: gen | judge | recog | explain
analysis.py        the three tables, computed from data/
prompts.jsonl      the 50 prompts and the computed answer keys
data/              answers, verdicts, recognition probes, explanations
results/RESULTS.md the full write-up with significance tests
```

`data/verdicts.jsonl` also contains 192 verdicts from a fifth judge used in an
earlier version of this experiment, an out-of-family model that was later
dropped from the panel. `analysis.py` filters them out. They are kept because
deleting real data to tidy a result is exactly the habit this experiment is
about, and because that judge turned out to have the worst position bias of
any model tested, which is worth knowing.

## Limitations

Forty-eight prompts, one prompt distribution, one rubric, one day, temperature
0. Gemini's reversed recognition split is unexplained. A planned test that
paraphrases every answer into a neutral style, to see whether the fingerprint
can be washed out, was attempted and abandoned: the paraphraser barely rewrote
the shortest answers, so the run proved nothing and is not included.

## Related work

This is a small independent replication of results that already exist in the
literature:

- Panickssery, Bowman and Feng, [LLM Evaluators Recognize and Favor Their Own
  Generations](https://arxiv.org/abs/2404.13076)
- Wataoka, Takahashi and Ri, [Self-Preference Bias in
  LLM-as-a-Judge](https://arxiv.org/abs/2410.21819), which links the preference
  to perplexity: judges reward text that is familiar to them.

## License

MIT for the code. The prompt set is derived from third-party datasets; check
their terms before redistributing.
