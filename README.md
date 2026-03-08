# PhilTrace

PhilTrace is a competition-ready philosophy analysis platform with two major capabilities:

1. Concept Drift Studio: analyze semantic change of philosophical concepts across historical periods.
2. Philosophy Chat: discuss ideas with local LLMs served by Ollama.

The project now includes a Bauhaus-inspired UI redesign and hardened backend reliability (structured errors, cache resilience, fallback behavior, tests).

## Features

- Concept input + historical definitions by period (Ancient / Medieval / Modern / Contemporary)
- Semantic scatter visualization (UMAP projection)
- Cluster evolution and drift trend charts
- Representative quotations with sources
- Chat interface with model selection and philosophy style toggle
- Backend request tracing (`X-Request-ID`) and consistent error responses
- Cache corruption recovery and fallback embedding mode

## Architecture

- `main.py`: Streamlit app entrypoint
- `ui/`: front-end pages and styles
  - `ui/view_drift.py`
  - `ui/view_chat.py`
  - `ui/style.py`
- `backend/`: FastAPI services
  - `backend/main.py`
  - `backend/analysis_engine.py`
  - `backend/llm_service.py`
  - `backend/data_models.py`
- `utils/`: API client and mock data for graceful fallback
- `corpus/`: period-based text corpus
- `tests/`: backend and engine tests

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_backend.txt
```

### 2. Start backend

```bash
uvicorn backend.main:app --reload
```

### 3. Start frontend

```bash
streamlit run main.py
```

### 4. Open app

- Frontend: http://localhost:8501
- Backend docs: http://localhost:8000/docs

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Environment Variables (optional)

- `PHILTRACE_LOG_LEVEL` (default: `INFO`)
- `PHILTRACE_BOOTSTRAP_ENGINE` (default: `true`)
- `PHILTRACE_ENABLE_CACHE` (default: `true`)
- `PHILTRACE_CACHE_TTL_SECONDS` (default: `604800`)
- `PHILTRACE_API_TIMEOUT_SECONDS` (default: `120`)
- `PHILTRACE_OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `PHILTRACE_HOST` (default: `0.0.0.0`)
- `PHILTRACE_PORT` (default: `8000`)

## License

MIT
