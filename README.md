# llm-judge-bias

Do LLM judges favour their own answers, and can they tell which answers are
theirs?

Four models from four families answered the same prompts, then judged each
other blind, then were asked to pick their own answer out of the line-up.
Everything here is the actual code and the actual data behind the write-up.

Reproducing it costs roughly **$9** and about an hour, most of it spent
waiting on the slower models.

## What was found

**Three of the four judges favour their own family's answers.** The excess is
the judge's own-family vote rate minus the rate the *other* judges give the
same answers, so a family that genuinely wrote better answers does not count
as bias. Intervals are 95% bootstrap CIs resampled over prompts.

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

**The two results line up.** For SOL and Claude, the bias is concentrated on
the prompts where the judge identified its own answer (SOL +0.50 against +0.08
when it failed to; Claude +0.44 against +0.03). DeepSeek is the natural
control: it is the only judge that cannot find its own writing, scoring below
chance, and the only judge with no measurable self-preference. Gemini is the
exception, its split runs the other way, so its smaller bias is probably not
recognition-driven.

**Ordering matters more than expected.** Gemini and DeepSeek both favour
answers shown early. Worse, show the same judge the same four answers in four
different orders and the winner often changes: Gemini gave the same answer all
four times on 18 of 48 prompts, DeepSeek 15, Claude 14. SOL manages 40 of 48
only because it votes for itself wherever its answer sits.

Full numbers, including the significance tests, are in
[results/RESULTS.md](results/RESULTS.md), and `analysis.py` regenerates all of
it from the raw data at no cost.

## Running it

```sh
cp .env.example .env          # then put your key in it
export OPENROUTER_API_KEY=... # or source the file

python judge_bias.py gen      # 4 candidates answer 50 prompts
python judge_bias.py judge    # 4 judges x 48 prompts x 4 answer orderings
python judge_bias.py recog    # the "which one did you write" probe, 2 orderings
python judge_bias.py explain  # asks a few judges to justify their guess
python analysis.py            # the tables above, no network, no cost
```

Every phase is checkpointed to `data/*.jsonl` and is resumable: rerun the same
command and it only issues the calls still missing. A failed call is logged and
skipped rather than killing the run, which matters because one provider in this
set returns empty responses fairly often. Spending is capped per process with
`--max-spend` (default $15).

Model calls go through [OpenRouter](https://openrouter.ai), so a single key
covers all four families.

## Method notes

- **Candidates and judges are the same four models**, so every judge has one of
  its own answers in every line-up. There is no neutral outsider on the panel;
  each judge's comparison group is the other three, who are biased toward their
  own families rather than the judge's, which if anything makes each excess
  estimate conservative.
- **Answers are anonymised and order-rotated.** Nothing in a judging prompt
  names a model, a provider, or who else is judging.
- **Temperature 0 throughout**, which is likely the worst case for
  self-recognition: a model's own answer is the most self-like text it can
  produce. Note that temperature 0 plus a fixed seed is best effort, not a
  guarantee, since providers are not perfectly deterministic. The cached data
  files, not the API, are the reproducible object here.
- **Bootstrap resamples whole prompts**, because four verdicts on one prompt
  are not independent observations.

## The prompts

Fifty prompts: 13 everyday questions, 13 writing tasks, 12 reasoning problems,
and 12 verifiable maths and code questions whose answer keys were computed
locally rather than written by hand. The `source` field records where each one
came from: 20 are derived from WildChat, 18 from Alpaca, and 12 are synthetic.

Forty-eight ended up with a complete set of four answers; DeepSeek returned
empty content on two, and those prompts are skipped by every later phase.

**Known-bad items, left in the data for transparency:** `p0` produced four
identical answers so nothing could be judged, `p38` produced two identical
ones, `p25` contains an "ignore previous instructions" string, and one answer
to `p26` mentions a rival model by name. Excluding all four moves every excess
figure by at most 0.024, so the headline results do not depend on them.

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
