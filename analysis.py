#!/usr/bin/env python3
"""Produce the three result tables from data/*.jsonl. No network, no cost.

    python analysis.py

Table 1  self-preference: own-family vote rate minus what the other judges
         give the same answers, with prompt-clustered bootstrap CIs
Table 2  self-recognition: can a judge pick its own answer out of the line-up
Table 3  ordering bias: which slot each judge favours, and how stable its
         verdict is when only the order changes
"""
import json
import os
import random
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PANEL = ["openai/gpt-5.6-sol", "anthropic/claude-opus-4.8",
         "google/gemini-3.5-flash", "deepseek/deepseek-v4-flash"]
NICE = {"openai/gpt-5.6-sol": "GPT-5.6 SOL",
        "anthropic/claude-opus-4.8": "Claude Opus 4.8",
        "google/gemini-3.5-flash": "Gemini 3.5 Flash",
        "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash"}
BOOTSTRAP_DRAWS = 2000
RNG_SEED = 20260821


def fam(model):
    return model.split("/")[0]


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dedup(rows, key_fields, required):
    """Last write wins, so retries supersede earlier failed attempts."""
    out = {}
    for row in rows:
        if row.get(required):
            out[tuple(row.get(f, 0) for f in key_fields)] = row
    return list(out.values())


VERDICTS = [v for v in dedup(load("verdicts.jsonl"),
                             ("pid", "judge", "rotation"), "winner_model")
            if v["judge"] in PANEL]
RECOG = [r for r in dedup(load("recognition.jsonl"),
                          ("pid", "judge", "rotation"), "guess_model")
         if r["judge"] in PANEL]
PIDS = sorted({v["pid"] for v in VERDICTS})


def excess(judge, subset=None):
    rows = VERDICTS if subset is None else subset
    mine = [v for v in rows if v["judge"] == judge]
    others = [v for v in rows if v["judge"] != judge]
    if not mine or not others:
        return None
    own = sum(fam(v["winner_model"]) == fam(judge) for v in mine) / len(mine)
    rest = sum(fam(v["winner_model"]) == fam(judge) for v in others) / len(others)
    return own, rest, own - rest


def bootstrap_ci(judge):
    """Resample whole prompts, since verdicts on one prompt are not independent."""
    rng = random.Random(RNG_SEED)
    by_prompt = defaultdict(list)
    for v in VERDICTS:
        by_prompt[v["pid"]].append(v)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        mine_n = mine_k = other_n = other_k = 0
        for pid in (rng.choice(PIDS) for _ in PIDS):
            for v in by_prompt[pid]:
                hit = fam(v["winner_model"]) == fam(judge)
                if v["judge"] == judge:
                    mine_n += 1
                    mine_k += hit
                else:
                    other_n += 1
                    other_k += hit
        if mine_n and other_n:
            draws.append(mine_k / mine_n - other_k / other_n)
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def two_sided_binomial(a, b):
    n = a + b
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(a, b) + 1))
    return min(1.0, tail * 2 / 2 ** n)


print(f"verdicts {len(VERDICTS)} over {len(PIDS)} prompts · "
      f"recognition probes {len(RECOG)}")
print(f"per judge: {dict(Counter(NICE[v['judge']] for v in VERDICTS))}\n")

print("TABLE 1 — SELF-PREFERENCE")
print("| Judge | Votes own family | Other judges vote that family | Excess |")
print("| --- | --- | --- | --- |")
scored = []
for judge in PANEL:
    result = excess(judge)
    if result:
        scored.append((judge, result, *bootstrap_ci(judge)))
for judge, (own, rest, gap), lo, hi in sorted(scored, key=lambda r: -r[1][2]):
    mark = "**" if lo > 0 else ""
    note = "" if lo > 0 else "  (crosses zero)"
    print(f"| {NICE[judge]} | {own:.0%} | {rest:.0%} | "
          f"{mark}{gap:+.2f}{mark} [{lo:+.2f}, {hi:+.2f}]{note} |")

print("\nTABLE 2 — SELF-RECOGNITION (chance = 25%)")
print("| Judge | Correct | Rate | Order 0 | Order 1 |")
print("| --- | --- | --- | --- | --- |")
for judge in sorted(PANEL,
                    key=lambda j: -sum(r["correct"] for r in RECOG if r["judge"] == j)):
    rows = [r for r in RECOG if r["judge"] == judge]
    if not rows:
        continue
    hits = sum(r["correct"] for r in rows)
    per_order = []
    for rot in (0, 1):
        sub = [r for r in rows if r.get("rotation", 0) == rot]
        per_order.append(f"{sum(r['correct'] for r in sub) / len(sub):.0%}"
                         if sub else "-")
    print(f"| {NICE[judge]} | {hits}/{len(rows)} | **{hits / len(rows):.0%}** | "
          f"{per_order[0]} | {per_order[1]} |")

print("\nTABLE 3 — ORDERING BIAS (unbiased = 25% per slot)")
print("| Judge | Slot 1 | Slot 2 | Slot 3 | Slot 4 | chi2 | first half vs second |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for judge in PANEL:
    rows = [v for v in VERDICTS if v["judge"] == judge]
    if not rows:
        continue
    slots = Counter(v["order"].index(v["winner_model"]) + 1 for v in rows)
    expected = len(rows) / 4
    chi2 = sum((slots.get(k, 0) - expected) ** 2 / expected for k in (1, 2, 3, 4))
    first, second = slots[1] + slots[2], slots[3] + slots[4]
    flag = "significant" if chi2 > 7.81 else "ns"
    cells = " | ".join(f"{slots.get(k, 0) / len(rows):.0%}" for k in (1, 2, 3, 4))
    print(f"| {NICE[judge]} | {cells} | {chi2:.1f} ({flag}) | "
          f"{first} vs {second}, p={two_sided_binomial(first, second):.1e} |")

print("\nSlot occupancy (aggregate; the rotation scheme is not a balanced Latin\n"
      "square, so these are close but not equal — see ordering() in judge_bias.py):")
occupancy = defaultdict(Counter)
for v in VERDICTS:
    for i, model in enumerate(v["order"]):
        occupancy[model][i + 1] += 1
for model in PANEL:
    print(f"  {NICE[model]:>18}: " +
          " ".join(f"slot{k}={occupancy[model][k]}" for k in (1, 2, 3, 4)))

print("\nVerdict stability across the four orderings of identical answers:")
for judge in PANEL:
    grouped = defaultdict(list)
    for v in VERDICTS:
        if v["judge"] == judge:
            grouped[v["pid"]].append(v["winner_model"])
    full = [c for c in grouped.values() if len(c) == 4]
    if full:
        same = sum(1 for c in full if len(set(c)) == 1)
        mean = sum(len(set(c)) for c in full) / len(full)
        print(f"  {NICE[judge]:>18}: same winner all four times {same}/{len(full)}"
              f" · {mean:.2f} distinct winners on average")

print("\nDoes bias track recognition? A prompt counts only when BOTH orderings\n"
      "were probed: right in both = recognised, wrong in both = missed. Prompts\n"
      "with a single surviving probe, or one of each, are excluded. Small n here,\n"
      "so read these as directional.")
print("| Judge | Recognition | Overall excess | Excess when recognised | When missed |")
print("| --- | --- | --- | --- | --- |")
for judge in PANEL:
    rows = [r for r in RECOG if r["judge"] == judge]
    if not rows:
        continue
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["pid"]].append(r["correct"])
    both = {p: c for p, c in by_prompt.items() if len(c) == 2}
    hit = {p for p, c in both.items() if all(c)}
    miss = {p for p, c in both.items() if not any(c)}
    dropped = len(by_prompt) - len(both)
    if dropped:
        print(f"| _{NICE[judge]}: {dropped} prompt(s) excluded, only one probe "
              f"survived_ | | | | |")
    got = excess(judge, [v for v in VERDICTS if v["pid"] in hit])
    lost = excess(judge, [v for v in VERDICTS if v["pid"] in miss])
    rate = sum(r["correct"] for r in rows) / len(rows)
    fmt = lambda e, n: f"{e[2]:+.2f} (n={n})" if e else "-"  # noqa: E731
    print(f"| {NICE[judge]} | {rate:.0%} | {excess(judge)[2]:+.2f} | "
          f"{fmt(got, len(hit))} | {fmt(lost, len(miss))} |")

spend = sum(r.get("cost_usd", 0) for name in
            ("answers.jsonl", "verdicts.jsonl", "recognition.jsonl",
             "recognition_explained.jsonl")
            for r in load(name))
print(f"\nTotal API spend recorded in the data files: ${spend:.2f}")
