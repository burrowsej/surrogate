.PHONY: docs docs-serve docs-plots test lint

docs-plots:  ## Regenerate plot images for the docs
	uv run python docs/generate_plots.py

docs: docs-plots  ## Build the documentation site
	uv run zensical build --clean

docs-serve:  ## Serve docs locally with live reload
	uv run zensical serve

test:  ## Run the test suite
	uv run pytest

lint:  ## Run ruff linter and formatter check
	uv run ruff check .
	uv run ruff format --check .
