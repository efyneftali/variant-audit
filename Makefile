.PHONY: test diagrams

test:
	python3 -m pytest tests/ -v

diagrams:
	plantuml docs/diagrams/*.puml -o ../img
