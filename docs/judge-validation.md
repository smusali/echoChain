# Judge validation

Human labels are the check on the judge. This file defines that check before any study labels exist.

## Label file

One row is one comparison. The columns are:

- seed id
- step
- phase
- model
- annotator verdicts, one per frozen claim id
- the judge's 1 to 5 score
- the judge's 1 to 10 score

When two annotators are available, each comparison has two verdict rows and two score rows. The comparison set should cover cycles, models, and phases, with at least 50 pairs.

## Gate

Agreement is computed on that table, stratified by cycle, model, and phase:

- Krippendorff's alpha at least 0.67 on verdicts
- weighted Cohen's kappa at least 0.67 on the 1 to 5 scores
- Spearman rho at least 0.70 on the 1 to 10 scores

`echochain.analyze` computes those three statistics from a table. A synthetic fixture in the tests locks the arithmetic.

These labels are not collected. No agreement is claimed.
