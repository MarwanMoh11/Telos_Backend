"""
add_memories.py
Run this anytime you want to add new memories to ChromaDB.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime, timezone

CHROMA_PATH = "./chroma_local"
USER_ID     = "marwan"

local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client   = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="telos_memories_local",
    embedding_function=local_ef,
    metadata={"hnsw:space": "cosine"},
)

# ── Add your memories here ──────────────────────────────────
new_memories = [
    {"id": "m7",  "text": "Going to the gym helped release stress before finals",        "eeg": "high_stress"},
    {"id": "m8",  "text": "Listening to music helped me focus during late night study",  "eeg": "low_focus"},
    {"id": "m9",  "text": "Talking to a friend made me feel less lonely and anxious",    "eeg": "high_stress"},
    {"id": "m10", "text": "Taking 10 minute breaks every hour improved my productivity", "eeg": "focused"},
    {"id": "m11", "text": "Skipping meals made my anxiety worse during exam week",       "eeg": "high_stress"},
    {"id": "m12", "text": "Goal: finish all assignments 2 days before deadline",         "eeg": "focused"},
]

# ── Insert (safe, no duplicates) ────────────────────────────
existing_ids = set(collection.get()["ids"])
to_add = [m for m in new_memories if m["id"] not in existing_ids]

if to_add:
    now_iso = datetime.now(timezone.utc).isoformat()
    collection.add(
        documents=[m["text"] for m in to_add],
        ids=[m["id"] for m in to_add],
        metadatas=[
            {
                "eeg":        m["eeg"],
                "user_id":    USER_ID,
                "created_at": now_iso,
            }
            for m in to_add
        ],
    )
    print(f"✅ Added {len(to_add)} new memories for '{USER_ID}'.")

# ── Migration (Migrate from old mahmoud ID) ──────────────────
old_data = collection.get(where={"user_id": "mahmoud"})
if old_data["ids"]:
    print(f"🔄 Migrating {len(old_data['ids'])} memories from 'mahmoud' to '{USER_ID}'...")
    collection.update(
        ids=old_data["ids"],
        metadatas=[{**m, "user_id": USER_ID} for m in old_data["metadatas"]]
    )
    print("✅ Migration complete.")
else:
    print("ℹ️ No old memories found for migration.")

if not to_add and not old_data["ids"]:
    print("ℹ️ All memories already exist, nothing added.")

# ── Show all memories in DB ─────────────────────────────────
all_data = collection.get(include=["documents", "metadatas"])
print(f"\n📦 Total memories in DB: {len(all_data['ids'])}")
for doc_id, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
    print(f"  [{doc_id}] ({meta.get('eeg')}) {doc}")
