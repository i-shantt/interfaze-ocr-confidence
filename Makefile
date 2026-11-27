PY ?= python3
FUNSD_URL = https://guillaumejaume.github.io/FUNSD/dataset.zip

.PHONY: check test report data collect analyze sheets synth collect-synth

check: test report

test:
	$(PY) -m unittest discover -s tests -t .

report:
	$(PY) -m ocr_trust.analyze --from-csv --out results

data:
	mkdir -p data
	curl -L -o data/funsd.zip $(FUNSD_URL)
	cd data && unzip -q -o funsd.zip && rm -rf __MACOSX

collect:
	$(PY) -m ocr_trust.client data/dataset/testing_data/images --out raw

analyze:
	mkdir -p sheets
	$(PY) -m ocr_trust.analyze --raw raw --annotations data/dataset/testing_data/annotations --out results --examples sheets/disagreements.json

sheets:
	$(PY) -m ocr_trust.sheets --examples sheets/disagreements.json --images data/dataset/testing_data/images --out sheets

synth:
	$(PY) -m ocr_trust.synth --out synth --pages-per-level 5

collect-synth:
	$(PY) -m ocr_trust.client synth/images --out raw_synth --run 1
	$(PY) -m ocr_trust.client synth/images --out raw_synth --run 2
