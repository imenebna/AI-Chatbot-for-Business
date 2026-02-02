# AI Business Chatbot (RAG-only)

This Streamlit app is a **RAG-only** chatbot: it retrieves and displays the most relevant paragraphs from a **local knowledge base document** (no generative model).

## Files
- `app-chatbot.py` — Streamlit app
- `requirements_chatbot.txt` — dependencies
- `ai_business_guide.md` — the local knowledge base (edit/replace this file to change what the chatbot knows)

## Install (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements_chatbot.txt
```

## Run
```powershell
python -m streamlit run app-chatbot.py
```

## Notes
- The guided buttons are **section-specific** to avoid retrieving the same passages for every click.
- You can optionally apply a manual **section filter** inside the app (Advanced settings).
