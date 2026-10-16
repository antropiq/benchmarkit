# Benchmarkit LLM

This project was entirely vibe coded to be compatible with llama.cpp / local inference backend.

With it you can benchmark several llm's using natural language questions that every body understand 😆

🚀 To get you started:

* Edit the `resources/questions.json` file to add other questions if needed (see note below)

> The 20 initial questions come from https://matchingpennies.com/hard_questions_for_llms/ thank's 🙂

* Edit the .env file (project root) to point to the LLM's you have

* Initialize the project

This project uses a [pyenv](https://github.com/pyenv/pyenv) virtualenv named `benchmarkit`
(see the gitignored `.python-version` file) — it auto-activates when you `cd` into the
project, so no `source .../activate` step is needed:

```bash
cd /path/to/benchmarkit
pip install -e ".[test]"
```

> Using plain venvs instead? `python -m venv .venv && source .venv/bin/activate && pip install -e ".[test]"` —
> but keep only **one** environment, pyenv shims otherwise shadow `.venv/bin`.

* Build the frontend (requires Node.js)

```bash
cd web
npm install
npm run build
```

* Run it

```bash
benchmarkit           # serves the API + UI on http://127.0.0.1:8000
```

### Develop with hot reload

Terminal 1 (API):

```bash
uvicorn src.api.server:app --reload
```

Terminal 2 (frontend dev server on :5173, proxies `/api` to :8000):

```bash
cd web && npm run dev
```

## Running tests

Install the test dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

If you have good questions to add please share them! 🙏
