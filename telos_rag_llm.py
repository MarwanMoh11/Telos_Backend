"""
telos_rag_llm.py
Full RAG → LLM pipeline: retrieves memories from ChromaDB, builds a prompt,
and sends it to the LLM (Llama local or Gemini online).
"""

import os
import requests
import chromadb
import json
import re
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from datetime import datetime, timezone
from agents import AGENTS, BASE_LUMI_INSTRUCTIONS, ROUTING_PROMPT
import uuid

# ─── Configuration ──────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_local"

# Llama (local/remote) config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
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

# ─── Orchestrator & Tone Engine: Local Semantic anchors ─────────────────────────
AGENT_EMBEDDINGS = {}
TONE_EMBEDDINGS = {
    "STRESSED": ["I'm overwhelmed", "hard", "pressure", "tired", "struggling", "can't focus", "bad day", "stressed", "anxious"],
    "POSITIVE": ["great", "fun", "I'm happy", "relaxed", "good move", "enjoying this", "confident", "easy", "doing well"]
}
TONE_ANCHORS = {}

def precompute_anchors():
    global AGENT_EMBEDDINGS, TONE_ANCHORS
    print("\n[Engine] Pre-computing semantic anchors...")
    
    # Agent Anchors
    for key, agent in AGENTS.items():
        if "utterances" in agent:
            embeddings = local_ef(agent["utterances"])
            AGENT_EMBEDDINGS[key] = np.array(embeddings)
            
    # Tone Anchors
    for tone, utterances in TONE_EMBEDDINGS.items():
        embeddings = local_ef(utterances)
        TONE_ANCHORS[tone] = np.array(embeddings)
        
    print("  ✅ Semantic anchors ready.")

# Initialize anchors at module load
precompute_anchors()


# ─── Humanity Upgrade State ─────────────────────────────────────────────────────
# This tracks the 'Single Brain' continuity across specialist swaps.
LAST_BRAIN_STATE = {
    "facet": "therapy",
    "thought": "I am just starting my day with the user. I should be warm and observant."
}

def get_life_threads(user_id: str) -> str:
    """
    Retrieve high-level user goals, deadlines, and persistent issues to pin as context.
    """
    print(f"     🔍 Scanning Life Threads (Goal Guardian Engine)...")
    # We focus on the big things: deadlines, major goals, and deep persistent issues.
    queries = ["assignment deadline due", "university goals", "family issues", "mental health", "long term projects"]
    all_threads = []
    
    for q in queries:
        mems = get_memories(user_id, q, n=1)
        if mems and mems[0]['similarity'] > 0.5:
            text = mems[0]['text']
            # FILTER: We want raw life info, not previous chat turns if possible
            # If the memory is a chat turn, try to extract the user's part or skip if it's too generic
            if "User asked:" in text:
                # Extract just the user's input if it's a chat memory
                match = re.search(r"User asked: (.*?) \| Telos replied:", text)
                if match:
                    text = match.group(1)
                else:
                    continue # Skip noisy chat logs
            
            if text not in all_threads and len(text) > 10:
                all_threads.append(text)
                print(f"        📍 Priority Thread: '{text[:50]}...' (Score: {mems[0]['similarity']})")
            
    if not all_threads:
        return "No active life threads or deadlines detected."
    
    return "\n".join(f"- {t}" for t in all_threads[:3])


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
def route_to_agent(message: str, history: list = None, eeg_state: str = None) -> str:
    """
    100% Local Semantic Router.
    Calculates cosine similarity between the query and agent 'anchor utterances'.
    """
    # Organized Interruption: If EEG stress is high, force handoff to Therapy Agent
    if eeg_state == 'high_stress':
        print("  ⚠️ [Orchestrator] High stress detected! FORCING handoff to Lumi-Zen (Therapy).")
        return "therapy"
        
    try:
        # Embed the current user message
        query_embedding = np.array(local_ef([message])[0])
        
        best_score = -1
        selected_agent = "therapy" # Fallback
        
        for key, anchors in AGENT_EMBEDDINGS.items():
            for anchor in anchors:
                # Calculate Cosine Similarity
                norm_q = np.linalg.norm(query_embedding)
                norm_a = np.linalg.norm(anchor)
                if norm_q == 0 or norm_a == 0: continue
                
                score = np.dot(query_embedding, anchor) / (norm_q * norm_a)
                
                if score > best_score:
                    best_score = score
                    selected_agent = key
        
        print(f"  🧠 [Orchestrator] Semantic Match: {AGENTS[selected_agent]['name']} (score: {best_score:.2f})")
        return selected_agent
        
    except Exception as e:
        print(f"  ❌ [Orchestrator] Semantic Routing failed: {e}")
        return "therapy"


def build_system_prompt(memories: list[dict], current_eeg: str = None, game_context: str = None, 
                        agent_key: str = "therapy", life_threads: str = "") -> str:
    """
    Build a Lumi system prompt enriched with retrieved memories, current EEG, and the 'Single Brain' context.
    """
    global LAST_BRAIN_STATE

    memory_lines = "\n".join(
        f"  {i}. [{m['eeg']}] {m['text']} (relevance: {m['similarity']})"
        for i, m in enumerate(memories, start=1)
    )

    eeg_info = f"CURRENT BRAIN STATE: {current_eeg or 'Unknown'}"

    # Internal Handover Thought
    internal_thought = f"[INTERNAL THOUGHT: {LAST_BRAIN_STATE['thought']}]"

    game_instruction = ""
    if game_context:
        game_instruction = f"""
GAME CONTEXT:
{game_context}
You can choose to comment on the game or continue the conversation. If you choose not to say anything, simply output exactly: [SILENCE]
"""

    agent = AGENTS.get(agent_key, AGENTS["therapy"])

    return f"""{BASE_LUMI_INSTRUCTIONS}

{internal_thought}

LIFE THREADS (Persistent Context - Use these for proactive curiosity):
{life_threads}

CURRENT ACTIVITY INSTRUCTIONS:
{agent['prompt']}

{eeg_info}
{game_instruction}

PERSONALIZATION:
- USE THESE MEMORIES to respond personally. If they explain who the user is or what they like, take them as FACT:
{memory_lines}

- If EEG data suggests high stress or low focus, acknowledge it gently and briefly.
- ALWAYS address the user's specific context provided in memories if relevant.
- You are their companion; don't act like a generic AI or mention you are switching facets."""


# ─── Step 3a: Call Llama via OpenAI-compatible API ──────────────────────────────
def call_llama(system_prompt: str, user_message: str, history: list[dict] = [],
               model: str = "gemma4:e2b", temperature: float = 0.7, max_tokens: int = 1024) -> str:
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
        
        if not data.get("choices"):
            print(f"  ⚠️ [Llama] Model returned no choices. Response: {data}")
            return ""
        
        content = data["choices"][0]["message"]["content"]
        if not content:
            print(f"  ⚠️ [Llama] Model returned empty content.")
            
        return content
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
             model: str = "gemma4:e2b", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Route to the active LLM provider (Llama or Gemini).
    """
    if LLM_PROVIDER == "gemini":
        return call_gemini(system_prompt, user_message, history, temperature, max_tokens)
    else:
        return call_llama(system_prompt, user_message, history, model, temperature, max_tokens)


# ─── Full Pipeline ──────────────────────────────────────────────────────────────
def ask_telos(user_id: str, query: str, eeg_state: str = None, history: list[dict] = [], game_context: str = None) -> str:
    """
    End-to-end: User query → RAG retrieves memories → LLM responds → Save memory.
    """
    global LAST_BRAIN_STATE
    print(f"\n{'='*60}")
    print(f"🧠 Telos Brain Pipeline (User: {user_id})")
    print(f"{'='*60}")
    print(f"  Query    : {query}")
    print(f"  EEG State: {eeg_state or 'not specified'}")
    
    # 1. Update Persistent Life Threads
    life_threads = get_life_threads(user_id)

    # 2. Orchestration - Determine which facet should handle this
    agent_key = route_to_agent(query, history, eeg_state)
    
    # If we are switching facets, we log it internally but don't show the user
    if agent_key != LAST_BRAIN_STATE["facet"]:
         print(f"  🧠 [Brain] Internal shift to facet: {agent_key}")

    # 3. Retrieve memories (RAG)
    print(f"\n[Step 1] Querying ChromaDB for user '{user_id}'...")
    memories = get_memories(user_id, query, eeg_state)
    print(f"  ✅ Found {len(memories)} matching memories.")
    
    # 4. Build prompt with Single-Brain awareness
    print("\n[Step 2] Building prompt with Humanity Engine...")
    print(f"     💭 Injecting Hidden Thought: '{LAST_BRAIN_STATE['thought'][:60]}...'")
    system_prompt = build_system_prompt(memories, eeg_state, game_context, agent_key, life_threads)

    # 5. Call LLM
    target = GEMINI_MODEL if LLM_PROVIDER == "gemini" else LLM_ENDPOINT
    print(f"\n[Step 3] Dispatching to {LLM_PROVIDER.upper()}...")
    print(f"     Active Facet: {AGENTS[agent_key]['name']}")
    answer = call_llm(system_prompt, query, history)
    print(f"  Response Preview: {answer[:60]}...")
    
    # 6. Update Internal Thought for next turn
    LAST_BRAIN_STATE["facet"] = agent_key
    LAST_BRAIN_STATE["thought"] = f"The user asked about '{query}'. My response was: '{answer[:50]}...'. I need to stay in this vibe."

    # Save memory
    if not game_context or "[SILENCE]" not in answer:
        save_memory(user_id, f"User asked: {query} | Telos replied: {answer}", eeg_state)

    return answer


# ─── Burnout Engine ─────────────────────────────────────────────────────────────
def calculate_burnout_index(user_id: str, eeg_state: str = None, history: list = None) -> dict:
    """
    Calculate a burnout index from 0.0 to 1.0 based on RAG history, live EEG, and recent conversation flow.
    """
    print(f"\n  ⚖️ [Difficulty Engine] Calculating Index for User: {user_id}")
    
    # 1. Base burnout from memories
    memories = get_memories(user_id, "stress burnout tired exhausted overwhelmed", n=5)
    base_burnout = 0.5
    if len(memories) >= 2:
        base_burnout += 0.2
        print(f"     >> Found {len(memories)} stressful memories. Base Burnout: {base_burnout}")

    # 2. Semantic Tone Analysis (100% Local)
    if history:
        user_msgs = [msg["content"] for msg in history if msg.get("role") == "user"][-2:]
        if user_msgs:
            print(f"     🔍 Analyzing tone for {len(user_msgs)} messages via Semantic Matching...")
            full_text = " ".join(user_msgs)
            try:
                
                query_embedding = np.array(local_ef([full_text])[0])
                best_tone = "NEUTRAL"
                best_tone_score = 0.4
                for t_key, anchors in TONE_ANCHORS.items():
                    for anchor in anchors:
                        score = np.dot(query_embedding, anchor) / (np.linalg.norm(query_embedding) * np.linalg.norm(anchor))
                        if score > best_tone_score:
                            best_tone_score = score
                            best_tone = t_key
                print(f"     🧠 Semantic Tone: '{best_tone}' (score: {best_tone_score:.2f})")
                
                if best_tone == "STRESSED":
                    base_burnout += 0.6  # Significant weight to override other factors
                    print(f"     ⚠️ Semantic stress detected. Burnout Index +0.6")
                elif best_tone == "POSITIVE":
                    base_burnout -= 0.3
                    print(f"     ✅ Semantic positivity detected. Burnout Index -0.3")
            except Exception as e:
                print(f"     ❌ Semantic Tone Match Failed: {e}")
                
    # 3. Real-time EEG modifiers
    if eeg_state == 'high_stress':
        base_burnout += 0.3
        print(f"     🔥 EEG State is High Stress. Burnout Index +0.3")
    elif eeg_state == 'focused':
        base_burnout -= 0.3
        print(f"     🎯 EEG State is Focused. Burnout Index -0.3")
    elif eeg_state == 'tired':
        base_burnout += 0.2
        print(f"     😴 EEG State is Tired. Burnout Index +0.2")
        
    # Clamp between 0.0 and 1.0
    base_burnout = max(0.0, min(1.0, base_burnout))
    recommended_difficulty = round(1.0 - base_burnout, 2)
    
    print(f"     🏁 FINAL SCORE -> Burnout: {round(base_burnout, 2)} | Difficulty: {recommended_difficulty}")
    
    return {
        "burnout_index": round(base_burnout, 2),
        "recommended_difficulty": recommended_difficulty
    }


# ─── Step 5: Generate Adaptive Daily Check-In ──────────────────────────────────
def generate_daily_checkin(user_id: str) -> list[str]:
    """
    Generate 3 adaptive questions based on user memories, formatted as JSON.
    """
    print(f"\n[Check-In] Generating daily check-in for '{user_id}'...")
    memories = get_memories(user_id, "stress focus goals feelings daily checkin", n=5)
    
    memory_lines = "\n".join(f"- {m['text']} (State: {m['eeg']})" for m in memories)
    
    system_prompt = f"""You are a mental health companion app.
Based on the following recent memories of the user, generate EXACTLY 3 highly personalized, short daily check-in questions to ask them right now.
Return ONLY a valid JSON array of strings. Do not use markdown blocks or any other text.
If there are no memories, just return 3 general well-being questions.

Memories:
{memory_lines}
"""
    answer = call_llm(system_prompt, "Generate 3 daily check-in questions as a JSON array of strings.")
    
    try:
        # Extract JSON array using regex in case LLM wraps it in markdown
        match = re.search(r'\[.*\]', answer, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
        else:
            questions = json.loads(answer)
            
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Not a valid JSON array or empty.")
            
        # Ensure we return strings
        questions = [str(q) for q in questions][:3]
        
        # Pad if less than 3
        while len(questions) < 3:
            questions.append("Is there anything else on your mind today?")
            
        return questions
    except Exception as e:
        print(f"  ❌ LLM JSON parse failed: {e}. Output was: {answer}")
        return [
            "How are you feeling right now?",
            "What is your main focus for today?",
            "Is there anything causing you stress?"
        ]


# ─── Entry Point for Testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Test a simple query
    ask_telos(
        user_id="marwan",
        query="My name is Marwan and I feel great today!",
        eeg_state="baseline",
        history=[]
    )
