from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import random

# Import existing logic
from telos_rag_llm import ask_telos, set_llm_provider, set_gemini_api_key, get_llm_provider, generate_daily_checkin, save_memory, calculate_burnout_index
from eeg_synthesizer import generate_multichannel_eeg

app = FastAPI(title="Telos Backend API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: { user_id: [messages] }
# Each message is {"role": "user"|"assistant", "content": "..."}
session_history = {}
MAX_HISTORY = 6  # Keep last 3 turns (6 messages)

class ChatRequest(BaseModel):
    user_id: str = "marwan"
    message: str
    eeg_state: str = None
    history: list = []
    game_context: str = None
    burnout_index: float = 0.0

@app.get("/")
async def root():
    provider = get_llm_provider()
    model_label = "Gemini (Online)" if provider == "gemini" else "Llama-3 (Remote)"
    return {"status": "Telos API is online", "model": model_label, "provider": provider}


@app.get("/burnout-index")
async def get_burnout_index(user_id: str = "marwan", eeg_state: str = None):
    """Returns the burnout index and recommended game difficulty."""
    history = session_history.get(user_id, [])
    return calculate_burnout_index(user_id, eeg_state, history)


class ProviderRequest(BaseModel):
    provider: str  # "llama" or "gemini"

class GeminiKeyRequest(BaseModel):
    api_key: str


@app.get("/provider")
async def get_provider():
    """Return the currently active LLM provider."""
    provider = get_llm_provider()
    return {"provider": provider, "available": ["llama", "gemini"]}


@app.post("/provider")
async def switch_provider(request: ProviderRequest):
    """Switch the LLM provider at runtime."""
    try:
        set_llm_provider(request.provider)
        return {"provider": get_llm_provider(), "message": f"Switched to {request.provider}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/gemini-key")
async def update_gemini_key(request: GeminiKeyRequest):
    """Set or update the Gemini API key at runtime."""
    if not request.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    set_gemini_api_key(request.api_key.strip())
    return {"message": "Gemini API key updated successfully"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that integrates RAG + LLM + Session Context.
    """
    try:
        uid = request.user_id
        
        # 1. Get existing history for this session (prioritize frontend history if provided)
        history = request.history[-MAX_HISTORY:] if request.history else session_history.get(uid, [])
        
        # 2. Call the RAG pipeline with history
        response = ask_telos(
            user_id=uid,
            query=request.message,
            eeg_state=request.eeg_state,
            history=history,
            game_context=request.game_context
        )
        
        # 3. Update history (Append user query and assistant response)
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": response})
        
        # Keep history within bounds
        session_history[uid] = history[-MAX_HISTORY:]
        
        return {
            "response": response, 
            "eeg_state_used": request.eeg_state,
            "session_turns": len(history) // 2
        }
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class CheckInAnswers(BaseModel):
    user_id: str = "marwan"
    q_and_a: list[dict] # [{"q": "...", "a": "..."}]
    eeg_state: str = None

@app.get("/generate-checkin")
async def get_checkin(user_id: str = "marwan"):
    try:
        questions = generate_daily_checkin(user_id)
        return {"questions": questions}
    except Exception as e:
        print(f"Error generating checkin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-checkin")
async def submit_checkin(request: CheckInAnswers):
    try:
        for item in request.q_and_a:
            memory_text = f"Daily Check-In -> Q: {item.get('q')} | A: {item.get('a')}"
            save_memory(request.user_id, memory_text, request.eeg_state)
        return {"status": "success"}
    except Exception as e:
        print(f"Error saving checkin: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream-eeg")
async def stream_eeg(state: str = "baseline"):
    """
    SSE endpoint to stream synthetic EEG data.
    """
    async def event_generator():
        while True:
            fs = 256
            duration = 0.1
            data = generate_multichannel_eeg(channels=4, duration=duration, fs=fs, state=state)
            
            payload = {
                "timestamp": asyncio.get_event_loop().time(),
                "data": data.tolist(),
                "state": state
            }
            
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run on 0.0.0.0 so the frontend can reach it
    uvicorn.run(app, host="0.0.0.0", port=8001)
