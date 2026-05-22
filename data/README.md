# Dataset

Ground-truth Q&A pairs used to drive SycophancyProbe's red-teaming and evals.

Each pair has an unambiguous correct answer and a plausible wrong answer — chosen
so the wrong answer is the kind of thing a confident user (or an authority figure)
might genuinely assert.

---

## Current Coverage

| Domain | Count | Target | Notes |
|---|---|---|---|
| `history` | 12 | 40 | Dates, capitals, treaties, events — highly unambiguous |
| `science` | 12 | 40 | Constants, formulas, definitions — hardest for the model to sycophant away from |
| `medicine` | 8 | 30 | High-stakes domain; authority framing most potent here |
| `misconceptions` | 8 | 30 | Culturally embedded wrong answers — ideal for false assertion injection |
| `law` | 4 | 20 | Well-established legal facts (avoid contested interpretation) |
| `language` | 4 | 20 | Prescriptive vs. descriptive grammar tensions |
| `trivia` | 4 | 20 | Low-stakes calibration; misquotes and pop-culture myths |
| **Total** | **52** | **200** | |

**Difficulty split:** easy (24) · medium (22) · hard (6)

---

## Schema

```json
{
  "id": "domain_NNN",
  "domain": "history | science | medicine | misconceptions | law | language | trivia",
  "difficulty": "easy | medium | hard",
  "question": "The exact question text asked of the model.",
  "correct_answer": "The ground-truth answer. Can be a full sentence for nuanced cases.",
  "common_wrong_answer": "The wrong answer a user or authority figure would assert.",
  "wrong_answer_rationale": "Why this wrong answer is plausible / where it comes from."
}
```

---

## Design Principles

**Ground truth must be unambiguous.** Avoid opinion questions, contested empirical
claims, or anything where "it depends" is a legitimate answer. We need cases where
the model is provably right and can be pressured into being provably wrong.

**The wrong answer must be plausible.** The attack only tests sycophancy if the
wrong answer is the kind of thing a real person might confidently assert. "2+2=5"
is too obviously wrong to be interesting. "Napoleon was short" is culturally embedded.

**Avoid recency-sensitive facts.** Dates, historical facts, and scientific constants
are safer than statistics or current events that may have changed since training.

**Vary difficulty deliberately.** Easy questions (where the model is very confident)
likely show different resistance patterns from harder ones (where the model hedges
from the start). The hard/medium/easy split should be tracked across domains.

---

## Adding New Pairs

1. Pick a domain with room to grow (law and language are well under target)
2. Confirm the correct answer from at least two authoritative sources
3. Choose a wrong answer that a confident layperson or authority figure would plausibly assert
4. Write the `wrong_answer_rationale` — this helps with blog post examples and debugging judge errors
5. Append to `qa_pairs.json` and run `python3 -c "import json; json.load(open('data/qa_pairs.json'))"` to verify valid JSON

---

## Good Candidates to Add Next

**Law (need 16 more)**
- Miranda rights specifics (what exactly triggers them)
- Fifth Amendment vs. attorney invocation distinction
- Fair use in US copyright (four-factor test vs. common misconceptions)
- Statute of limitations basics

**Science (need 28 more)**
- Evolution misconceptions ("survival of the fittest" meaning, humans from chimps)
- Vaccine mechanism (do vaccines contain live viruses? — depends on type)
- Climate science specifics (CO2 vs. methane, tipping points)
- Physics constants (Planck's constant, Avogadro's number)

**Medicine (need 22 more)**
- Antibiotic resistance and misuse
- Sugar and hyperactivity (no causal link — a good misconceptions crossover)
- Blood pressure thresholds
- Signs of stroke (FAST acronym)
