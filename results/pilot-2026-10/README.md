# Pilot, October 2026

The planned run was 10 study seeds, three arms (`alternate`, `swap`, and `same-model`), and three repeats. That is 90 runs and 5,400 calls. A dry run before any chat call estimated that maximum and a cost upper bound of $14.054400, pricing 1,024 tokens on both the prompt side and the completion side of every call.

Five runs finished. They spent $0.15444. OpenRouter then returned HTTP 402: the account could reserve only 693 completion tokens, and each call asks for up to 1,024. The same-model arm and the other nine seeds were not run. The chart below is only the finished fragment.

![Mean fidelity for the finished Redis runs](fidelity.svg)

`alternate` is three repeats of `redis-epoll`. Mean fidelity stayed at 10 through step 16 and was 9.33 at step 30. `swap` is two repeats of the same seed, and they do not agree: one stayed near 10, and the other fell as low as 2, so the swap mean is not a stable estimate. `plot.py` draws the mean fidelity of the committed JSONL. Running it again on this directory produced the same SVG.

These runs use one judge and a 1 to 10 score. They do not split claims, they do not report a confidence interval, and they run for a fixed 30 steps. They are not the ten-seed comparison.
