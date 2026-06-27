# Makefile — Targeted ABSA Inference API
# Run `make` (or `make help`) to list available targets.
#
# IMPORTANT: Recipe lines use TAB indentation (Makefile requirement).
#
# dev: Phase 3 will add frontend dev server target here

PYTHON  := python3
UVICORN := uvicorn

.PHONY: api api-prod install help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

api: ## Start API dev server with hot-reload on port 8000
	PYTHONPATH=. $(UVICORN) api.main:app --reload --port 8000

api-prod: ## Start API for production — single worker, no reload
	PYTHONPATH=. $(UVICORN) api.main:app --workers 1 --host 0.0.0.0 --port 8000

install: ## Install Python inference dependencies (CPU torch + requirements.txt)
	pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
