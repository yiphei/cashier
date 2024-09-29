.PHONY: format lint typecheck

format:
	isort .
	black .

lint:
	flake8 .

typecheck:
	mypy .

check: format lint typecheck
