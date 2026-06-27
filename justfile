# Targeted ABSA Demo

# Install all deps
install:
    uv sync
    cd frontend && npm install

# Start backend API (port 8000)
api:
    PYTHONPATH=. uv run uvicorn api.main:app --workers 1 --port 8000

# Start frontend dev server (port 5173)
dev:
    cd frontend && npm run dev
