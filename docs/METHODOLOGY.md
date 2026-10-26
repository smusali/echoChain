# Methodology

echoChain measures how a technical statement changes as models rewrite it. This note is the measurement definition for rubric version 2. A finished run can be rebuilt from its manifest: package version, prompt template hashes, corpus hash, condition, models, temperatures, `top_p`, and `sampling_seed`.

## What is measured

The primary metric is claim fidelity. Each corpus seed has a frozen list of claims. The judge marks every claim `supported`, `contradicted`, or `not_mentioned`. Claim fidelity is the share marked `supported`. The contradiction rate is the share marked `contradicted`. Neither ratio is compared with the stop threshold.

Every judge call also returns two holistic scores. `anchored` is a 1 to 5 score whose levels separate a preserved mechanism from a preserved topic. `score` is the existing 1 to 10 comparison. The 1 to 10 mean is the only number the stop rule reads. Question steps are scored for answerability and stay off the fidelity curves.

A later answer is also judged against the previous answer, not against the question in between. The first answer leaves those fields empty. Seed-targeted samples and previous-answer samples are stored in separate lists.

## Conditions

`alternate` and `swap` are telephone chains with rotating models. `same-model` keeps the telephone prompts and uses one model. `identity` asks the model to repeat the text. `paraphrase` asks for a paraphrase and does not turn the text into a question. Repeats multiply whole runs. The command default is one repeat. `examples/study.toml` sets three.

## Judge panel

The default judge is `openai/gpt-4.1-mini`, with 3 samples at temperature 0.5. A model family is the slug before the first slash. A run is rejected when a judge family is also a generator family. Model ids are not written into the prompts.

## Sampling

`seed` in the configuration selects a corpus item. The sampling integer is `sampling_seed`. When `--sampling-seed` is omitted, each run draws one positive integer and stores it on the manifest. A call sends `base + step * 1000 + sample_index * 10 + target`. Target 0 is the transform, 1 is the seed comparison, and 2 is the previous-answer comparison. The row records that integer. `top_p` defaults to 1.0, is sent on every call, and is stored on the manifest and on the step parameters. `seed_honored` is true or false only when the provider echoes a seed. Otherwise it is null.

## Summaries

`echochain analyze` pools runs only when they share one rubric version. A missing new field is treated as null. Curve intervals resample seed ids with `random.Random(0)`, keep every run of a drawn seed, and use a linear percentile. A run with no seed id is its own cluster. The survival event is the first step whose 1 to 10 mean is at or below the manifest threshold. If that never happens, the time is the last completed step and the event flag is 0. That includes stops for budget, generation, judge failure, and interrupt. A censored observation at time `t` stays in the risk set at `t` and then leaves. The Kaplan-Meier curve is the product limit. Pairwise log-rank tests use the Mantel-Haenszel variance and a two-sided p-value on one degree of freedom.

## Limitations

- Judge family bias. The vendor check keeps the judge's family off the generator list, and the human-label protocol in `docs/judge-validation.md` is the remaining check. Those labels are not collected.
- Provider nondeterminism. The client sends a seed, records whether the provider echoed it, repeats runs, and reports clustered intervals.
- Model retirement. Catalog preflight rejects an unknown id, and the nightly smoke job calls the configured models.
- Small English-only corpus. The held-out split is the unused portion of this corpus. The runs do not support a claim about a broader population.
- Answerability. Question steps stay off the fidelity curves.
- Injected instructions. Chain text is wrapped in tags, and a colliding tag is shielded before it is sent.
