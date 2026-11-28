# Results

Interfaze-1, OCR task, 50 FUNSD test forms. 8191 predicted words were scored against the labels; 317 had no labelled box under them and 40 were checkboxes, both left out. 42 labelled words had no predicted word over them.

**Lenient** ignores case and punctuation. **Strict** compares exact text after Unicode normalisation.

| | lenient | strict |
|---|---|---|
| word accuracy vs. labels | 95.8% | 91.8% |
| mean stated confidence | 94.3% | 94.3% |
| ECE, 10 bins (lower is better) | 0.023 [0.017, 0.032] | 0.040 [0.032, 0.053] |
| Brier score | 0.041 | 0.070 |
| AUROC, confidence separates right from wrong | 0.781 [0.733, 0.827] | 0.760 [0.727, 0.794] |
| words with confidence ≥ 0.995 | 14.1% | 14.1% |
| their accuracy vs. labels | 98.9% | 98.1% |
| document mean confidence vs. document error rate (Spearman) | -0.62 | -0.48 |

Brackets: 95% intervals from resampling whole documents (1,000 resamples).

## Reliability (lenient)

| stated confidence | words | mean confidence | accuracy vs. labels |
|---|---|---|---|
| 0.00–0.50 | 134 | 35.2% | 66.4% |
| 0.50–0.70 | 347 | 61.5% | 81.6% |
| 0.70–0.80 | 287 | 74.2% | 84.0% |
| 0.80–0.90 | 404 | 85.3% | 89.1% |
| 0.90–0.95 | 349 | 92.5% | 94.3% |
| 0.95–0.99 | 1317 | 96.8% | 96.7% |
| 0.99–1.00 | 5353 | 99.2% | 98.4% |

## After checking disagreements by hand

130 randomly sampled items were checked against the image crop (`adjudication.csv`); 8 of them (checkboxes, `_X_`) are no longer disagreements under the current matching rules and are not counted.

| band | words | words wrong vs. labels | groups checked (forms) | OCR wrong | label wrong | ambiguous | artifact | estimated real accuracy (95% range) |
|---|---|---|---|---|---|---|---|---|
| conf ≥ 0.99 | 5353 | 84 (1.6%) | 37 (23) | 8 | 27 | 2 | 0 | 99.6% (99.2%–99.9%) |
| conf < 0.9 | 1172 | 199 (17.0%) | 39 (23) | 14 | 17 | 6 | 2 | 91.7% (85.1%–95.9%) |

Predicted words below 0.9 with **no label under them** (190 words, mean confidence 0.52) are left out of the scores above. 30 were checked: 18 real text read correctly, 2 misread, 6 specks or lines read as characters, 4 ambiguous: estimated 60.0% correct (36.7%–90.9%). Counting them, the whole band below 0.9 (1362 words, mean stated confidence 0.67) is an estimated **87.3%** correct (78.3%–95.2%).

Estimated real accuracy counts words: (disagreement groups in the band / groups checked) × (words in checked groups judged OCR errors), divided by the band's words. Ambiguous cases count as correct for the estimate and as errors for the low end of the range. Ranges resample whole forms (2,000 resamples), because several checked items can come from one form. Words between 0.90 and 0.99 were not hand-checked.

Disagreements in case or punctuation only: 16 checked, 16 were label conventions or label errors, 0 were OCR errors, 0 ambiguous. That is why lenient matching is the headline number.

## Using confidence to route words to review (lenient, vs. labels)

Accept words at or above a threshold; send the rest to a person. Measured against the labels, so it overstates the review needed.

| target accuracy | threshold | words sent to review | errors caught | a perfect detector would send |
|---|---|---|---|---|
| 98% | 0.93 | 16.6% | 60.7% | 2.3% |
| 99% | – | – | – | 3.3% |
| 99.5% | – | – | – | 3.7% |

## Output defects found

- **Duplicated text when a word is split in two:** flagged automatically 9 times in 2 of 50 forms. The detector is conservative: the hand check found one more it misses (`37` read as `3737`). One printed word comes back as two boxes and the second, narrow box repeats the first box's text, so joining the words gives e.g. "NONO.:" for "NO.:". Examples: "NO" (1.00) + "NO.:" (0.89), "NO" (0.96) + "NO.:" (0.88), "NO" (0.99) + "NO.:" (0.83).
- **Checkboxes:** returned as `[ ]` ×28, `[x]` ×6, `I` ×1, `x` ×1, `B` ×1, `xx` ×1, `[` ×1, `[]` ×1.

Box quality: median IoU between matched predicted and labelled word groups is 0.76; 96.2% are at IoU ≥ 0.5.
