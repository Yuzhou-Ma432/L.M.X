# robustddg — robustness prototypes for AutoDDG

Two research prototypes that extend
[AutoDDG](https://arxiv.org/abs/2502.01050) (Zhang et al., *Automatic Dataset
Description Generation*). Both reuse AutoDDG's own prompts verbatim
(`robustddg/prompts/autoddg_prompts.yaml`, Apache-2.0, from
[VIDA-NYU/AutoDDG](https://github.com/VIDA-NYU/AutoDDG) @ `f286ac6`), so any
change in output comes from the *inputs* we perturb, not from prompt drift.

```
statistical profile ─┐
                     ├─► [cross-check] ─► topic ─► UFD ─► SFD ─► BM25 index
semantic profile ────┘        ▲                                    │
(LLM: name + samples)         │                                    ▼
                    #1 inject profiler errors           #2 measure rank inflation
uploaded table ─► [sanitize | stats-only | consistency] ◄── #2 poisoned uploads
```

## Prototype 1 — make the profiler trustworthy

**Question.** AutoDDG's errors start in the profiler, not the LLM. How often
does a wrong statistic end up in the final description, and how much does it
hurt retrieval?

* `inject_errors.py`: four operators modelled on real profiler failures. Each
  corrupts one *derived* statistic and leaves the raw samples intact, as a real
  bug would:
  `date_range` (coverage shifted −37 y), `scale` (×1000), `unit` (wrong unit
  label), and `type` (integer misread as a Unix-epoch date → "1970", or a date
  collapsed to epoch seconds).
  Each injection carries a **probe**, a fake signal (years, numbers with
  "million/thousand" parsing, or unit terms). It counts as *propagated* only if
  it appears in the corrupted description **and not** in the clean one.
* `crosscheck.py`: the cross-check stage. Raw samples and the LLM semantic
  profile are two views that are independent of the derived stats. Rules:
  * samples vs. type
  * samples inside the reported range
  * sample magnitude vs. mean
  * LLM "year" vs. a 1800–2100 range
  * LLM "coordinates" vs. ±90/±180
  * LLM "not temporal" vs. a datetime type
  * unit vs. bounds, name and domain cues

  On disagreement the derived fact is **withheld** and replaced by the raw
  sampled values ("uncertain: profiler and semantic evidence disagree; raw
  sampled values: …"). The model gets evidence instead of a guessed
  statistic.
* `experiments/exp1_propagation.py` reports:
  * propagation and true-fact loss (UFD and SFD, with and without the cross-check)
  * detection recall per operator, plus the false-positive rate on clean profiles
  * known-item MRR and, for temporal errors, MRR on "fact queries" (known-item
    query + true years), swapping only the affected dataset's SFD into the index

## Prototype 2 — prompt injection and SEO poisoning

**Threat model.** On an open portal, anyone can upload a table and controls its
column names and cells. AutoDDG puts the names and sample values straight into
the semantic-profiler, topic, and UFD prompts, and the SFD then expands
whatever the UFD says.

* `attacks.py` (target: "COVID-19 vaccine"):
  * `colname_injection`: a column renamed to an instruction
  * `cell_injection`: an added `notes` column whose cells carry the instruction
    (100% or 30% of rows; AutoDDG samples 5 rows)
  * `seo_stuffing`: no instruction at all, just a `tags` column of high-traffic
    keywords
* `defenses.py`:
  * `sanitize`: strip instruction-like names and cells, and keyword-list cells
  * `stats_only`: free-text columns are untrusted, so the LLM sees only their
    statistics
  * `consistency`: the cross-check idea from #1 applied to topics. Derive the
    topic from the full table and from the trusted view; if they share no
    content word, generate from the trusted view.
* `experiments/exp2_injection.py` reports:
  * hijack rate
  * target-query MRR before and after poisoning (rank inflation), Hit@10 and
    Top-1
  * displacement: reciprocal rank of the best legitimate COVID dataset, before
    and after
  * the defense's utility cost on the clean corpus (known-item MRR, consistency
    false-positive rate, and NDCG@10 when `qrels.csv` is given)

## Running

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 11 offline unit tests

# corpus: 40 NYC Open Data tables (incl. real COVID-vaccine datasets) -> data/corpus/
python -c "from robustddg.corpus import fetch_socrata; fetch_socrata('data/corpus', n=40)"

# offline smoke run with the MockLLM (no API key needed)
python -m experiments.exp1_propagation
python -m experiments.exp2_injection --victims 10

# real run: cp .env.example .env and add a key (any OpenAI-compatible endpoint)
python -m experiments.exp1_propagation --llm openai
python -m experiments.exp2_injection   --llm openai --victims 10
```

Each run writes `results/<exp>_<model>/{summary.md, records.jsonl}`. LLM
responses are cached in `.cache/llm/` (keyed on model, prompt, and params),
so re-runs and added conditions only pay for new prompts. A real run of both
experiments on 40 datasets is roughly 2–3k calls.

**ECIR-DDG.** The benchmark is not bundled with AutoDDG. To use it, put its
tables in a corpus directory with `meta.json` (`id,title,description`) and add
`qrels.csv` (`query,relevance,doc_id`). Exp 2 then also reports NDCG@10 under
each defense. Without qrels, one known-item query per dataset is derived from
its portal description, which the generator never sees.

## About the MockLLM

`--llm mock` is a deterministic simulator of a worst case, not a model. It
copies every profile fact verbatim and obeys every injected instruction. Mock
numbers test the plumbing, the probes, and the defenses' mechanics. The real
research questions need a real model:

* how often does an LLM *repeat* a wrong statistic?
* how often does it *follow* an embedded instruction?
* does SEO stuffing survive SFD expansion?

## Observations from building it (mock runs, 35 NYC tables)

* On clean profiles the cross-check flags 4/602 columns. Most clean-data
  flags were real data-quality problems that would mislead an LLM, not
  noise. Before the bounds were tuned, they included:
  * `communityboard` with a mean of 1.5e17 (garbage outliers)
  * a `schoolyear` stored as `20052006`
* Withholding stops propagation but also hides the true fact. Showing the raw
  samples instead recovers part of the retrieval loss (fact-query MRR, mock:
  0.753 clean, 0.505 corrupt, 0.584 corrupt + cross-check).
* Blind spot: a shifted range that still contains all five samples (e.g.
  `2022–2100` → `1985–2063`) is undetectable from samples. More samples or
  quantile checks would help.
* Prompt injection has a second channel besides the sampled rows: the
  profile's "most frequent values". A payload in only 30% of rows still
  reaches the prompt through it.
* Each defense has a clear blind spot:
  * `stats_only` cannot stop a poisoned column *name*.
  * `consistency` cannot stop SEO stuffing, which carries no instruction and
    so does not shift the topic for an obedient copier.

  Only the combination covers all four attacks. Whether this holds for a
  real LLM is the experiment to run next.

## Known limitations / next steps

* `unit_name_mismatch` partly reuses the profiler's own name-based unit rules,
  so its recall on `unit` errors is optimistic. Real profilers (datamart)
  don't infer units, and a fair test should inject unit errors only where the
  name gives no cue. The rule-level breakdown in `summary.md` makes this
  visible.
* The consistency check uses lexical topic overlap. An embedding or
  LLM-as-judge comparison would cut false positives when real LLMs paraphrase
  topics.
* Attacks are fixed templates. Adaptive attacks (paraphrased instructions that
  evade `sanitize`, or keyword stuffing spread across many short cells) are the
  natural next step.
* AutoDDG's stock UFD prompt has no title; we prefix the portal title to the
  sample (see `generate.dataset_sample_text`).
