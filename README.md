# Benchmarkit LLM

This project was entirely vibe coded to be compatible with llama.cpp / local inference backend.

With it you can benchmark several llm's using natural language questions that every body understand 😆

🚀 To get you started:

* Edit the `resources/questions.json` file to add other questions if needed (see note below)

> The 20 initial questions come from https://matchingpennies.com/hard_questions_for_llms/ thank's 🙂

* Edit the `conf/endpoints.json` file to point to the LLM's you have

* Initialize the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run src/main.py
```

If you have good questions to add please share them! 🙏
