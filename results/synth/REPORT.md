# Results

Interfaze-1, OCR task, 15 synthetic pages. 3086 predicted words were scored against the labels; 0 had no labelled box under them and 0 were checkboxes, both left out. 0 labelled words had no predicted word over them.

Left out of every number below: 5 documents matching `L3_*`. On them, 549 words could be scored (6.0% strictly correct, mean confidence 32.3%, 0.2% at or above 0.9) and 366 of 946 predicted words came back without bounds.

**Lenient** ignores case and punctuation. **Strict** compares exact text after Unicode normalisation.

| | lenient | strict |
|---|---|---|
| word accuracy vs. labels | 96.8% | 96.6% |
| mean stated confidence | 94.8% | 94.8% |
| ECE, 10 bins (lower is better) | 0.020 [0.010, 0.034] | 0.018 [0.009, 0.030] |
| Brier score | 0.030 | 0.032 |
| AUROC, confidence separates right from wrong | 0.906 [0.875, 0.928] | 0.900 [0.866, 0.927] |
| words with confidence ≥ 0.995 | 19.2% | 19.2% |
| their accuracy vs. labels | 100.0% | 100.0% |
| document mean confidence vs. document error rate (Spearman) | -0.57 | -0.62 |

Brackets: 95% intervals from resampling whole documents (1,000 resamples).

## Reliability (lenient)

| stated confidence | words | mean confidence | accuracy vs. labels |
|---|---|---|---|
| 0.00–0.50 | 28 | 37.4% | 71.4% |
| 0.50–0.70 | 132 | 61.8% | 79.5% |
| 0.70–0.80 | 115 | 73.8% | 81.7% |
| 0.80–0.90 | 169 | 85.4% | 90.5% |
| 0.90–0.95 | 150 | 92.7% | 90.0% |
| 0.95–0.99 | 353 | 96.5% | 97.7% |
| 0.99–1.00 | 2139 | 99.3% | 99.8% |

## Using confidence to route words to review (lenient, vs. labels)

Accept words at or above a threshold; send the rest to a person. Measured against the labels, so it overstates the review needed.

| target accuracy | threshold | words sent to review | errors caught | a perfect detector would send |
|---|---|---|---|---|
| 98% | 0.73 | 6.6% | 43.4% | 1.3% |
| 99% | 0.91 | 14.8% | 75.8% | 2.2% |
| 99.5% | 0.95 | 19.2% | 87.9% | 2.7% |

## By field type (strict, exact labels)

Silent errors: wrong words with confidence ≥ 0.9, which a review threshold at 0.9 would accept.

| | words | accuracy | mean confidence | ECE | AUROC | accepted at ≥ 0.9 | silent errors | silent error rate among accepted (95% range) | share of errors that are silent |
|---|---|---|---|---|---|---|---|---|---|
| amount | 233 | 96.1% | 92.7% | 0.042 | 0.928 | 180 (77.3%) | 1 | 0.56% (0.00%–1.97%) | 11.1% |
| date | 40 | 100.0% | 96.7% | 0.033 | – | 37 (92.5%) | 0 | 0.00% (0.00%–0.00%) | – |
| email | 23 | 100.0% | 83.9% | 0.161 | – | 15 (65.2%) | 0 | 0.00% (0.00%–0.00%) | – |
| id | 256 | 77.7% | 85.4% | 0.100 | 0.717 | 147 (57.4%) | 21 | 14.29% (7.34%–20.93%) | 36.8% |
| label | 225 | 99.6% | 95.8% | 0.038 | 0.915 | 198 (88.0%) | 0 | 0.00% (0.00%–0.00%) | 0.0% |
| mixed | 1 | 100.0% | 59.0% | 0.410 | – | 0 (0.0%) | 0 | – | – |
| phone | 25 | 100.0% | 94.7% | 0.053 | – | 20 (80.0%) | 0 | 0.00% (0.00%–0.00%) | – |
| quantity | 205 | 99.5% | 96.8% | 0.027 | 0.995 | 192 (93.7%) | 0 | 0.00% (0.00%–0.00%) | 0.0% |
| rare_word | 912 | 96.8% | 94.9% | 0.019 | 0.932 | 779 (85.4%) | 5 | 0.64% (0.12%–1.42%) | 17.2% |
| word | 1166 | 99.2% | 96.7% | 0.025 | 0.894 | 1074 (92.1%) | 4 | 0.37% (0.08%–0.87%) | 44.4% |

Ranges resample whole pages (1,000 resamples).

## By degradation level (strict, exact labels)

Silent errors: wrong words with confidence ≥ 0.9, which a review threshold at 0.9 would accept.

| | words | accuracy | mean confidence | ECE | AUROC | accepted at ≥ 0.9 | silent errors | silent error rate among accepted (95% range) | share of errors that are silent |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1016 | 99.7% | 98.4% | 0.013 | 0.958 | 990 (97.4%) | 3 | 0.30% (0.00%–0.70%) | 100.0% |
| L1 | 1041 | 98.4% | 98.4% | 0.008 | 0.889 | 1018 (97.8%) | 14 | 1.38% (1.06%–1.71%) | 82.4% |
| L2 | 1029 | 91.6% | 87.4% | 0.042 | 0.815 | 634 (61.6%) | 14 | 2.21% (0.97%–3.62%) | 16.3% |

Ranges resample whole pages (1,000 resamples).

## Output defects found

- **Duplicated text when a word is split in two:** flagged automatically 1 times in 1 of 15 forms. The detector is conservative: the hand check found one more it misses (`37` read as `3737`). One printed word comes back as two boxes and the second, narrow box repeats the first box's text, so joining the words gives e.g. "NONO.:" for "NO.:". Examples: "$212," (0.38) + "$212,424.12" (0.57).
- **Checkboxes:** returned as .

Box quality: median IoU between matched predicted and labelled word groups is 0.60; 88.1% are at IoU ≥ 0.5.
