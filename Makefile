.PHONY: test diagrams eval

test:
	python3 -m pytest tests/ -v

diagrams:
	plantuml docs/diagrams/*.puml -o ../img

# make eval                    -> full run against all golden_dataset.jsonl rows
# make eval ARGS="--limit 5"   -> smoke test on the first 5 rows before a full run
eval:
	python3 evals/run_evals.py $(ARGS)
