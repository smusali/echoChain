# Pre-registration

This document freezes the study design before the runs start. The manifest stores the SHA-256 of this file. A later edit does not change a run that already recorded the hash.

## Hypotheses

- Telephone arms (`alternate` and `swap`) lose claim fidelity across answer steps.
- `same-model` keeps a higher horizon claim fidelity than `alternate`.
- Technical-token survival and cosine to the seed move with claim fidelity.
- `generic_share` rises across answer steps.

## Sample

The corpus is `seeds/v1.toml`, version `v1`, split `study` (10 seeds). Held-out seeds `utf8-encoding` and `consistent-hashing` are excluded. The conditions are `alternate`, `swap`, and `same-model`, with three repeats: 90 runs. That is the sample. It is the whole study split, not a power calculation, and it is above a minimum of 3 runs for each of 3 seeds.

## Models and prompts

Generators are `google/gemini-3.8-flash`, `deepseek/deepseek-v4.1-flash-1017`, and `mistralai/mistral-small-2603`. The judge is `openai/gpt-4.1-mini`, with 3 samples at temperature 0.5. The rubric version is 2. The transform temperature is 0.3, `top_p` is 1.0, `max_tokens` is 1024, and `max_cycles` is 30.

## Providers

There are no endpoint pins. Calls set `require_parameters` and `allow_fallbacks` false. The served provider is recorded on the manifest. `data_collection` and `zdr` stay unset.

## Stop rule

The stop rule is `fixed_length` because `stop_consecutive` is 0. The survival event is still the answer step whose 1 to 10 mean first reaches 2.0 or below. A stored consecutive of 0 is read as 1. Horizon `tau` is 30. The primary metric is claim fidelity. The 1 to 10 mean is only the stop and survival input.

## Baseline

Answer steps store a `baseline` object that does not use the judge. Cosine similarity and entailment use pinned models: embedding `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, and NLI `cross-encoder/nli-MiniLM2-L6-H768` at revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`. Forward entailment treats the reference as the premise and the answer as the hypothesis. The stored value is the entailment-class probability from the model's own label map. `length_ratio` is the answer's Unicode code points divided by the seed's. An empty seed has no ratio. `technical_survival` is the share of the seed's technical tokens that occur again as whole tokens. Technical tokens are casefolded complexity expressions with internal spaces removed, digit literals, acronyms of two or more capitals, hyphenated compounds, and remaining alphabetic tokens of length at least 3 that are not in a fixed closed-class list. `generic_share` is the share of the answer's word tokens that are in that list. Question steps store the same object with every value null. The first answer leaves the previous-answer fields null.

A pre-registered run refuses to start when the `baseline` extra is not installed, so cosine and entailment are not omitted.

## Analysis

Analysis uses the current `echochain analyze` rules: one rubric, nulls stay null, seed-clustered `Random(0)` intervals, Kaplan-Meier, restricted mean, and log-rank. Baseline curves are sanity checks, not extra primary endpoints. For `generic_share` and `length_ratio`, the report prints an ordinary least-squares fit of the last value on the first when a condition has at least three runs with both endpoints and the absolute slope is below 1. The printed strength is `1 - slope`, and the position is the intercept divided by that strength. Otherwise the report prints that the fit was not estimated.

## Exclusions

Errored judge samples stay out of the mean. Censored runs stay in the survival sample. A credit stop is resumed rather than dropped. Mixed rubrics are not pooled.

## Agreement

The agreement gate is unmet. It requires Krippendorff's alpha at least 0.67, weighted kappa at least 0.67, and Spearman rho at least 0.70. Those labels are not collected. No agreement is claimed.

## Timestamp

An optional timestamp on OSF or AsPredicted is not part of this file. The hash stored on the manifest is the record the study uses.
