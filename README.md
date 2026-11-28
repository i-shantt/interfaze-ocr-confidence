# Sure and wrong: Interfaze's OCR confidence on ID codes

Interfaze's homepage suggests using its OCR confidence to *"send low-confidence values to review instead of trusting them blindly"* ([interfaze.ai](https://interfaze.ai)), and its `llms.txt` says the raw OCR in `precontext` *"is what you want for auditing or confidence thresholds."* I tested that advice on pages where the right answer for every word is known exactly, and on real scanned forms.

For ordinary words the advice works. For ID codes (invoice, account and policy numbers, the fields document extraction exists to capture) it doesn't. Interfaze is confidently wrong about look-alike characters, it makes the same mistake every time, and the fix only needs what is already in its response.

![Confident errors](results/synth/confident_errors.png)

*Every word Interfaze got wrong at confidence ≥ 0.9 on the exact-label pages. Left: the page as scanned. Right: the printed text and what came back, with the differences in red.*

## Findings

1. **Confidence is trustworthy for words.** On the exact-label pages, 10 of the 2,495 non-code words accepted at confidence ≥ 0.9 are wrong (0.4%). Calibration error is 0.018. On real scanned forms (FUNSD), words at ≥ 0.99 are about 99.6% right once label errors are taken out by hand.

2. **It is not trustworthy for codes.** On the same pages, **21 of the 147 ID codes accepted at ≥ 0.9 are wrong (14%; page-resampled range 7–21%)**, about 35 times the rate for other words. Even at ≥ 0.99, 4 of 76 codes are wrong, against 1 of 2,063 other words. 28 of the 31 confident errors are swaps between look-alike characters (`0`/`O`/`o`, `1`/`I`/`l`, `5`/`S`, `8`/`B`). The same kind of error shows up on real scans: `lowa` for *Iowa* and `Retall` for *Retail*, both at 0.99.

3. **The printed text was visible in the pixels.** On every noisy page, the swapped characters differ by at least 5.3 standard deviations of that page's own noise (d′, explained below). An observer who knew the font would make each of these swaps less than 0.4% of the time. Three more errors are on noise-free pages, where the image is exactly the printed text.

4. **Asking twice does not help, and the score does not describe the text it is attached to.** All 31 confident errors came back identically on a second run. Across 4,770 words sent twice, confidence never changed between runs, not even for the 24 words whose text did. `Scattle` and `Seattle` both scored 0.61 on the same word. Two readings of one word cannot both be right, so 0.61 cannot be the chance that each is right.

5. **A fix with three numbers.** I recalibrated confidence using two facts already in the response: whether a word mixes letters and digits, and what share of its characters have look-alikes. At the same review budget as a 0.95 threshold, this **cuts confident code errors from 11 to 3** (fitted and tested on different pages). Fitted only on synthetic pages and applied to real FUNSD forms, it also ranks errors better there: AUROC 0.781 → 0.807, gain 0.011–0.041.

## What I'd change in Interfaze

- **Make confidence aware of look-alike characters in codes.** Lower the score when a token mixes letters and digits and contains characters that have look-alikes. Better still, return a per-character confidence or the runner-up character, so a code's weakest position is visible. Today a code's score looks like a word's score, but it is far less reliable.
- **Say what the score means.** It stays the same when the text changes between runs, so it reflects something like the image region rather than the returned string. Users choosing a review threshold need to know that.
- **Smaller output fixes:**
  - Don't repeat text when one printed word is split into two boxes (`NO` + `NO.:` → `NONO.:`).
  - Return checkboxes in one documented format (I saw 8: `[ ]`, `[x]`, `[]`, `[`, `I`, `x`, `B`, `xx`).
  - Keep returning boxes on very degraded pages: on the worst synthetic pages, 366 of 946 words came back with no bounds.

Two developer-experience points:
- No response header shows the keyless daily quota. The `X-RateLimit-*` headers track the 50 requests/second limit, so the first sign that the 50-per-day quota is gone is an error.
- `llms.txt` tells agents *"Interfaze requires an API key to work"*, but the [Limits](https://interfaze.ai/docs/limits) page and FAQ say keyless calls work.

## Objections I expected

**"Synthetic pages aren't real documents."**
They are used where exact labels are needed: real datasets have label errors (FUNSD's do, see below), and those errors swamp a 1% effect. Two checks against real scans support the synthetic result:
- The same look-alike errors appear in FUNSD's top confidence band.
- The calibrator, fitted only on synthetic pages, improves error ranking on real forms it never saw.

The pages use seven common business fonts at three degradation levels, from a clean render to an office scan.

**"Your IDs are adversarial: random strings full of look-alikes."**
On purpose, because that is where OCR goes wrong, and real codes contain these characters. VINs leave out I, O and Q for exactly this reason; most invoice, policy and order numbers don't.

The claim is about confidence, not accuracy. However hard codes are, the score should say so, and it doesn't: 14% of confident codes are wrong against 0.4% of confident words. Some of my codes contain a lowercase `l`, which real all-caps codes rarely do. Leaving those out, **16 of 141 confident codes are still wrong (11%)**.

**"Some of those characters are identical in some fonts. Georgia's 0 looks like an o."**
I measured it instead of arguing it. For each confident error, the printed character and the returned one are drawn alone, in the word's font and size, and put through the page's blur, downscale and contrast. The returned character may slide 3 px sideways to fit best, so width and position earn no credit. The difference is divided by the noise measured on that page, giving d′. The smallest is 5.3; Georgia's `0` read as `o` scores 40–54. Even a truly ambiguous character should get about 50% confidence, not 0.99.

**"Your labels are wrong" / "you checked them yourself."**
The synthetic labels come from the code that drew the text: every word's text and box is known exactly. A test checks that on one page per level, rotated pages included, no ink falls outside the label boxes and every box contains ink. For FUNSD, all 130 hand-check verdicts are published with form and box, so anyone can dispute them.

**"15 pages is small."**
Every range resamples whole pages, not words. The ranges for codes (7–21%) and words (0.1–0.9%) are far apart. More pages would tighten them; the keyless API allows 50 calls a day, and these took two days of quota.

**"`precontext` is the raw OCR; the model's final answer is corrected."**
`precontext` is where the confidence lives, and Interfaze's own `llms.txt` points there for thresholds. I didn't test whether the structured output repairs codes. That depends on the prompt, and it carries no score to threshold on.

**"You tuned the calibrator on FUNSD."**
No. All four feature sets I tried were fitted only on synthetic pages, and FUNSD was used only to score them. All four are reported below, including the 12-feature model, which does well on synthetic pages but worse than doing nothing on FUNSD (0.765). That failure is why the proposed model is the small one.

**"The keyless endpoint might not be the production model."**
Every one of the 100 responses reports `"model": "interfaze-1"`. All calls were made on 27 November 2026 (US Pacific time).

## Details

### Exact labels, by field type

15 synthetic pages (levels 0–2), 3,086 words, each scored against its exact label. A *silent error* is a wrong word at or above the review threshold, so nobody looks at it. Ranges resample whole pages. Full tables: [`results/synth/REPORT.md`](results/synth/REPORT.md).

| field | words | accuracy | mean confidence | accepted at ≥ 0.9 | silent errors | wrong among accepted |
|---|---|---|---|---|---|---|
| ID code | 256 | 77.7% | 85.4% | 147 | 21 | **14.3%** (7.3–21.0%) |
| amount | 233 | 96.1% | 92.7% | 180 | 1 | 0.6% (0–2.0%) |
| common word | 1,166 | 99.2% | 96.7% | 1,074 | 4 | 0.4% (0.1–0.9%) |
| rare word | 912 | 96.8% | 94.9% | 779 | 5 | 0.6% (0.1–1.4%) |
| field label | 225 | 99.6% | 95.8% | 198 | 0 | 0% |
| quantity | 205 | 99.5% | 96.8% | 192 | 0 | 0% |
| date, phone, email | 88 | 100% | 92.8% | 72 | 0 | 0% |

By level: clean renders are 99.7% right, light scans 98.4%, office scans 91.6%, with calibration error 0.008–0.042.

Level-3 pages (a bad fax at about 67 dpi) are left out of every number. Interfaze reads only 6% of their words correctly, but it says so: mean confidence is 0.32, and 0.2% of words reach 0.9. It returns 366 of their 946 words with no bounds, so word-level scoring is not meaningful there.

### The confident errors

[`results/synth/confident_errors.csv`](results/synth/confident_errors.csv) lists all 31, with the edits, font, d′ and whether run 2 repeated them.
- 21 are ID codes, 5 rare words, 4 common words and 1 amount (`$84.07` read as `$84,07`).
- Only 3 are not look-alike swaps: `eorthkin`, `tabor`, `closcal`.

How d′ is computed:
1. Draw each printed character and the character returned in its place on their own, in the word's font and size.
2. Apply the page's blur, downscale and contrast to both.
3. Let the returned character slide up to 3 px sideways to its best fit.
4. Sum the squared pixel differences over the swapped characters to get S.
5. Measure the page's own noise, σ: re-render the whole page without noise ([`synth.make_page(clean=True)`](ocr_trust/synth.py)) and compare it with the real crop. JPEG loss counts as noise, so σ is if anything too large.

Then d′ = √S / σ. An observer who knew the font and had to choose between the two characters would choose wrongly with probability Φ(−d′/2). The median d′ is 29 and the minimum 5.3.

### Repeat runs

[`results/synth/consistency.json`](results/synth/consistency.json), [`results/funsd/consistency.json`](results/funsd/consistency.json)

| | words paired across runs | same confidence | text changed | changed, same confidence > 0.5 |
|---|---|---|---|---|
| synthetic, levels 0–2 | 3,086 | 100% | 16 | 12 |
| FUNSD, 10 forms | 1,684 | 100% | 8 | 8 |

All the text changes are below 0.9, so a 0.9 threshold already sends them to review. Repeat-and-compare is a cheap guard a user might add, and it would not catch a single one of the 31 confident errors.

### The fix

[`results/synth/routing.json`](results/synth/routing.json). Each policy reviews exactly as many words as a plain threshold would (14.4% of words for 0.9, 19.2% for 0.95), so only which words get reviewed changes.

Synthetic results are out of fold: 5 folds by page, so no page is scored by a model fitted on it. FUNSD results come from models fitted on all synthetic pages and on no FUNSD data, scored against FUNSD's labels with case and punctuation ignored. AUROC gain ranges resample whole pages.

| model | synthetic AUROC | silent errors (codes), budget of 0.9 | budget of 0.95 | FUNSD AUROC (gain) |
|---|---|---|---|---|
| raw confidence | 0.900 | 31 (21) | 14 (11) | 0.781 |
| Platt scaling | 0.897 | 31 (21) | 14 (11) | 0.781 (0) |
| + letters and digits | 0.912 | 22 (10) | 9 (6) | 0.785 (−0.000 to 0.010) |
| **+ look-alike share** | **0.922** | **22 (7)** | **9 (3)** | **0.807 (0.011 to 0.041)** |
| all 12 features | 0.916 | 21 (6) | 11 (3) | 0.765 (−0.035 to 0.002) |

The proposed model, fitted on all 3,086 synthetic words ([`results/synth/calibrator.json`](results/synth/calibrator.json)):

```
p(correct) = σ(1.49 + 0.81·logit(confidence) − 0.79·[mixes letters and digits] − 1.61·(share of look-alike characters))
```

It lowers a code stated at 0.95 with 60% look-alike characters to 0.89. It raises an ordinary word stated at 0.90 to 0.96, which matches FUNSD, where the low end of the scale understates accuracy. To apply it to any Interfaze OCR response:

```bash
python -m ocr_trust.calibrate apply results/synth/calibrator.json response.json   # confidence, p(correct), word
```

### Real forms: FUNSD

The 50 test forms of [FUNSD](https://guillaumejaume.github.io/FUNSD/) (noisy business scans from the 1980s and 90s) with word-level labels: 8,191 predicted words scored. The labels contain errors, so I hand-checked 130 disagreements against the image. Full tables: [`results/funsd/REPORT.md`](results/funsd/REPORT.md); verdicts: [`results/funsd/adjudication.csv`](results/funsd/adjudication.csv).

![FUNSD reliability](results/funsd/reliability.png)

- **Top band.** 65% of words come back at ≥ 0.99, and 98.4% of them match the labels. Of 37 hand-checked disagreements there, 27 were label mistakes (the page says `638-6532` where the label has `638-5532`), 8 real OCR errors and 2 ambiguous. That puts real accuracy at an estimated **99.6%** (99.2–99.9%), against a stated 99.2%.
- **Below 0.9 the scale understates accuracy.** These 1,362 words average 0.67 confidence but are an estimated **87% right** (78–95%), counting the words with no label under them, which were checked separately.
- **Ranking.** AUROC 0.78 [0.73, 0.83]. A form's mean confidence tracks its error rate (Spearman −0.62).
- **Duplicated split text.** A printed `NO.:` comes back as `NO` (0.99) followed by a narrow box reading `NO.:`. It is flagged 9 times in 2 forms, always with the duplicate box below 0.9.
- **Strict comparison against FUNSD mostly measures label conventions.** All 16 sampled case- or punctuation-only disagreements were label conventions or label errors. That is why FUNSD numbers ignore case and punctuation.

## How it works

- **Matching** ([`match.py`](ocr_trust/match.py)). Interfaze and the labels split text differently (`Fax:614-466-5087` can be one word or two), so predicted and labelled boxes that overlap by at least half of the smaller box are joined into groups. A group is right when its text, read in row order, matches. Words returned without bounds can't be placed and aren't scored.
- **Synthetic pages** ([`synth.py`](ocr_trust/synth.py)). Seeded and byte-for-byte reproducible. Each page has a header, a key-value block, a 12–15 row line-item table and notes, every token tagged by kind. Level 0 is a clean 150 dpi render. Level 1 adds a light scan (blur, 75% scale, noise, JPEG 70). Level 2 adds an office scan (55% scale, up to 1° rotation, faded contrast, JPEG 50). The images and labels are committed.
- **Metrics** ([`metrics.py`](ocr_trust/metrics.py)). ECE, Brier, AUROC with tied ranks, risk–coverage, and bootstraps that resample whole documents. Standard library only.

## Run it

Needs Python 3.10+ and nothing else; matplotlib is optional and only draws the reliability figures.

```bash
make check    # tests, plus rebuilding every published number from the committed CSVs (no API calls)
```

CI runs this on Python 3.10 and 3.13 and fails if any committed result changes. The committed CSVs hold confidences, boxes, features and right/wrong labels, but no OCR text.

To reproduce from scratch (the keyless API allows 50 calls a day; the client caches responses and stops cleanly at the limit, so rerun the next day; set `INTERFAZE_API_KEY` to use a key):

```bash
make data collect collect-run2 analyze      # FUNSD: download, 50 + 10 calls, score
make collect-synth analyze-synth            # synthetic pages: 40 calls, score; needs Pillow and the macOS fonts
make sheets                                 # crops of sampled disagreements, for hand checking
```

## Limitations

- **One run date, one model** (`interfaze-1`, keyless), and English, printed text only.
- **15 exact-label pages.** The ranges allow for it, but the code result rests on 147 confidently accepted codes.
- **The ID generator is mine.** It over-represents look-alike characters on purpose. The silent-error *rate* on real codes depends on how often they contain such characters; the *gap* between codes and words is the finding.
- **One person did the FUNSD hand check**, and words between 0.90 and 0.99 weren't checked. Lenient matching counts `1.5` and `15` as equal.
- **The calibrator has been tested on two datasets.** Users with their own labelled pages should refit it: `python -m ocr_trust.calibrate fit`.
- **Raw API responses are not included**, because Interfaze's terms restrict redistributing output. Short quoted errors are included for discussion. FUNSD is for non-commercial research use and is downloaded from its source, not redistributed.

Not affiliated with Interfaze.
