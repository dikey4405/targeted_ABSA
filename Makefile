# Makefile — Targeted ABSA Demo

.PHONY: api dev install help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install all deps (uv sync + npm install)
	uv sync
	cd frontend && npm install

api: ## Start backend API (port 8000)
	PYTHONPATH=. uv run uvicorn api.main:app --workers 1 --port 8000

dev: ## Start frontend dev server (port 5173)
	cd frontend && npm run dev
