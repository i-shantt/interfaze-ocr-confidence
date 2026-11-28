PY ?= python3
FUNSD_URL = https://guillaumejaume.github.io/FUNSD/dataset.zip
FUNSD = data/dataset/testing_data
# Level-3 synthetic pages (about 67 dpi) are below what OCR is expected to read, and Interfaze returns many of
# their words without boxes, so word-level scoring is not meaningful there. They are summarised separately.
SYNTH_EXCLUDE = 'L3_*'

.PHONY: check test report data collect collect-run2 analyze sheets synth collect-synth analyze-synth

# ---- no API calls, no dependencies: rebuild every published number from the committed CSVs ----

check: test report

test:
	$(PY) -m unittest discover -s tests -t .

report:
	$(PY) -m ocr_trust.analyze --from-csv --out results/funsd
	$(PY) -m ocr_trust.analyze --from-csv --out results/synth --name "synthetic pages" --exclude $(SYNTH_EXCLUDE)
	$(PY) -m ocr_trust.calibrate evaluate --train results/synth/word_features.csv \
		--transfer results/funsd/word_features.csv --out results/synth/routing.json
	$(PY) -m ocr_trust.calibrate fit --train results/synth/word_features.csv --set lookalike --out results/synth/calibrator.json

# ---- FUNSD: 50 calls for run 1, 10 more for the repeat run ----

data:
	mkdir -p data
	curl -L -o data/funsd.zip $(FUNSD_URL)
	cd data && unzip -q -o funsd.zip && rm -rf __MACOSX

collect:
	$(PY) -m ocr_trust.client $(FUNSD)/images --out raw

collect-run2:
	$(PY) -m ocr_trust.client $(FUNSD)/images --out raw --run 2 --limit 10

analyze:
	mkdir -p sheets
	$(PY) -m ocr_trust.analyze --raw raw --annotations $(FUNSD)/annotations --out results/funsd --examples sheets/disagreements.json
	$(PY) -m ocr_trust.calibrate features --raw raw --labels $(FUNSD)/annotations --out results/funsd/word_features.csv
	$(PY) -m ocr_trust.consistency raw --out results/funsd/consistency.json

sheets:
	$(PY) -m ocr_trust.sheets --examples sheets/disagreements.json --images $(FUNSD)/images --out sheets

# ---- synthetic pages: 20 pages x 2 runs = 40 calls ----

synth:
	$(PY) -m ocr_trust.synth --out synth --pages-per-level 5

collect-synth:
	$(PY) -m ocr_trust.client synth/images --out raw_synth --run 1
	$(PY) -m ocr_trust.client synth/images --out raw_synth --run 2

analyze-synth:
	$(PY) -m ocr_trust.analyze --raw raw_synth --annotations synth/labels --out results/synth --name "synthetic pages" \
		--exclude $(SYNTH_EXCLUDE)
	$(PY) -m ocr_trust.calibrate features --raw raw_synth --labels synth/labels --exclude $(SYNTH_EXCLUDE) \
		--out results/synth/word_features.csv
	$(PY) -m ocr_trust.consistency raw_synth --exclude $(SYNTH_EXCLUDE) --out results/synth/consistency.json
	$(PY) -m ocr_trust.lookalike --raw raw_synth --labels synth/labels --images synth/images --exclude $(SYNTH_EXCLUDE) \
		--out results/synth
