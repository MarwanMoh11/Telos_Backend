# Telos Backend (RAG & LLM Provider)

This is the FastAPI backend for the Telos application. It handles user chat interactions, maintains conversation memory via ChromaDB, and serves a synthetic EEG stream for the frontend visualizer.

## Architecture
- **FastAPI**: Core server framework.
- **ChromaDB**: Local vector database for Retrieval-Augmented Generation (RAG).
- **Dual-LLM Support**: Supports routing queries to local open-weight models (Llama/Gemma) or cloud models (Google Gemini).

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   python run.py
   ```
   (Alternatively, use `uvicorn main:app --reload --port 8001`)

3. **API Endpoints:**
   - `POST /chat`: Submit a user message and receive an AI response.
   - `GET /stream-eeg`: SSE endpoint for synthetic brainwave data.
   - `POST /provider`: Switch between LLM providers (llama, gemini).
   - `POST /gemini-key`: Inject a Gemini API key securely at runtime.
