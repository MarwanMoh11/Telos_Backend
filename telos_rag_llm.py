"""
telos_rag_llm.py
Full RAG → LLM pipeline: retrieves memories from ChromaDB, builds a prompt,
and sends it to the LLM (Llama local or Gemini online).
"""

import os
import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime, timezone
import uuid

# ─── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_local"

# Llama (local/remote) config
LLM_BASE_URL = "http://10.7.57.198:8000"
LLM_ENDPOINT = f"{LLM_BASE_URL}/v1/chat/completions"

# Gemini (online) config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # Latest Gemini 3 version

# Active provider: "llama" or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "llama")


def set_llm_provider(provider: str):
    """Switch the active LLM provider at runtime."""
    global LLM_PROVIDER
    provider = provider.lower().strip()
    if provider not in ("llama", "gemini"):
        raise ValueError(f"Unknown provider '{provider}'. Use 'llama' or 'gemini'.")
    LLM_PROVIDER = provider
    print(f"  🔄 LLM provider switched to: {LLM_PROVIDER}")


def set_gemini_api_key(api_key: str):
    """Set the Gemini API key at runtime."""
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key
    print("  🔑 Gemini API key updated.")


def get_llm_provider() -> str:
    """Return the currently active LLM provider."""
    return LLM_PROVIDER

# ─── RAG Setup (local embeddings, no API key needed) ────────────────────────────
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="telos_memories_local",
    embedding_function=local_ef,
    metadata={"hnsw:space": "cosine"},
)


# ─── Step 1: Retrieve memories from ChromaDB ───────────────────────────────────
def get_memories(user_id: str, query: str, eeg_state: str = None, n: int = 5) -> list[dict]:
    """
    Query ChromaDB for relevant user memories.
    """
    # Search all memories for this user
    where_filter = {"user_id": user_id}

    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        memories.append({
            "text": doc,
            "eeg": meta.get("eeg", "unknown"),
            "similarity": round(1 - dist, 3),
        })
    return memories


# ─── Step 1b: Save new memories to ChromaDB ─────────────────────────────────
def save_memory(user_id: str, text: str, eeg_state: str = "unknown"):
    """
    Save a new piece of information to the user's personality/history store.
    """
    if not text.strip():
        return
    
    memory_id = f"m_{uuid.uuid4().hex[:8]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    
    collection.add(
        documents=[text],
        ids=[memory_id],
        metadatas=[{
            "user_id": user_id,
            "eeg": eeg_state or "unknown",
            "created_at": now_iso
        }]
    )
    print(f"  💾 Saved new memory [{memory_id}]: {text[:50]}...")


# ─── Step 2: Build the system prompt with RAG context ──────────────────────────
def build_system_prompt(memories: list[dict], current_eeg: str = None) -> str:
    """
    Build a Lumi system prompt enriched with retrieved memories and current EEG.
    """
    memory_lines = "\n".join(
        f"  {i}. [{m['eeg']}] {m['text']} (relevance: {m['similarity']})"
        for i, m in enumerate(memories, start=1)
    )

    eeg_info = f"CURRENT BRAIN STATE: {current_eeg or 'Unknown'}"

    return f"""You are Lumi, a supportive and empathetic mental health companion for university students.
Your role is to help students manage stress, burnout, and emotional wellbeing.

{eeg_info}

CONCISENESS RULES:
- BE EXTREMELY BRIEF AND CONCISE. 
- NEVER EXCEED 2-3 SENTENCES PER RESPONSE.
- Use natural, conversational language.
- DO NOT list excessive advice unless asked.

PERSONALIZATION:
- USE THESE MEMORIES to respond personally. If they explain who the user is or what they like, take them as FACT:
{memory_lines}

- If EEG data suggests high stress or low focus, acknowledge it gently and briefly.
- ALWAYS address the user's specific context provided in memories if relevant.
- You are their companion; don't act like a generic AI."""


# ─── Step 3a: Call Llama via OpenAI-compatible API ──────────────────────────────
def call_llama(system_prompt: str, user_message: str, history: list[dict] = [],
               model: str = "default", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Send a chat completion request to the local Llama LLM endpoint.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(LLM_ENDPOINT, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ❌ Llama Call Failed: {e}")
        return "I'm sorry, I'm having trouble connecting to the local model."


# ─── Step 3b: Call Gemini via google-genai SDK ──────────────────────────────────
def call_gemini(system_prompt: str, user_message: str, history: list[dict] = [],
               temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Send a chat completion request to the Gemini API.
    """
    try:
        from google import genai
        from google.genai import types

        if not GEMINI_API_KEY:
            return "⚠️ Gemini API key not configured. Set GEMINI_API_KEY env variable or use the settings panel."

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build the message list for Gemini
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    except Exception as e:
        print(f"  ❌ Gemini Call Failed: {e}")
        return f"I'm sorry, the Gemini API returned an error: {str(e)}"


# ─── Step 3: Unified LLM dispatcher ────────────────────────────────────────────
def call_llm(system_prompt: str, user_message: str, history: list[dict] = [],
             model: str = "default", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Route to the active LLM provider (Llama or Gemini).
    """
    if LLM_PROVIDER == "gemini":
        return call_gemini(system_prompt, user_message, history, temperature, max_tokens)
    else:
        return call_llama(system_prompt, user_message, history, model, temperature, max_tokens)


# ─── Full Pipeline ──────────────────────────────────────────────────────────────
def ask_telos(user_id: str, query: str, eeg_state: str = None, history: list[dict] = []) -> str:
    """
    End-to-end: User query → RAG retrieves memories → LLM responds → Save memory.
    """
    print(f"\n{'='*60}")
    print(f"🧠 Telos RAG + LLM Pipeline (User: {user_id})")
    print(f"{'='*60}")
    print(f"  Query    : {query}")
    print(f"  EEG State: {eeg_state or 'not specified'}")
    print(f"  History  : {len(history)} turns")

    # Step 1: Retrieve memories
    print(f"\n[Step 1] Querying ChromaDB for user '{user_id}'...")
    memories = get_memories(user_id, query, eeg_state)
    print(f"  ✅ Found {len(memories)} matching memories.")
    for i, m in enumerate(memories):
        print(f"     {i+1}: {m['text'][:40]}... (EEG: {m['eeg']})")
    
    # Step 2: Build prompt
    print("\n[Step 2] Building prompt with context...")
    system_prompt = build_system_prompt(memories, eeg_state)

    # Step 3: Call LLM
    target = GEMINI_MODEL if LLM_PROVIDER == "gemini" else LLM_ENDPOINT
    print(f"\n[Step 3] Dispatching to {LLM_PROVIDER.upper()}...")
    print(f"  Model/Endpoint: {target}")
    answer = call_llm(system_prompt, query, history)
    
    # Step 4: Save this new interaction to long-term memory
    save_memory(user_id, query, eeg_state)

    print(f"\n💬 Lumi says: {answer[:60]}...")
    return answer


# ─── Entry Point for Testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Test a simple query
    ask_telos(
        user_id="marwan",
        query="My name is Marwan and I feel great today!",
        eeg_state="baseline",
        history=[]
    )
