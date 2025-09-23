from datetime import datetime
from transformers import pipeline
import os
from dotenv import load_dotenv
import asyncio

# Load Hugging Face token if needed
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==============================================================================
print("🚀 Initializing HuggingFace pipelines with GPU acceleration...")

# Use smaller, faster models and enable GPU (device=0)
try:
    summarizer_pipeline = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Smaller model
        device=0,
        torch_dtype="auto"
    )
    print("✅ Summarizer pipeline ready")
except Exception as e:
    print("❌ Error loading summarizer:", e)
    raise

try:
    action_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",  # Smaller model
        device=0,
        torch_dtype="auto"
    )
    print("✅ Action pipeline ready")
except Exception as e:
    print("❌ Error loading action extractor:", e)
    raise

# ==============================================================================
def preprocess_chats(chats):
    print("🧹 Preprocessing chats...")
    cleaned = []
    for c in chats:
        if c.get("message") and c['message'].strip():
            try:
                ts = datetime.fromisoformat(c['timestamp'])
            except Exception:
                ts = datetime.now()
            cleaned.append({
                "timestamp": ts,
                "user": c.get("user", "Unknown"),
                "message": c['message'].strip()
            })
    print(f"✅ {len(cleaned)} messages cleaned")
    return cleaned

# ==============================================================================
async def summarize(conversation_text):
    result = summarizer_pipeline(conversation_text, max_length=100, min_length=30, truncation=True)
    return result[0]['summary_text']

async def extract_actions(conversation_text):
    prompt = f"Extract action items with deadlines and owners:\n{conversation_text}"
    result = action_pipeline(prompt, max_length=100, truncation=True)
    return result[0]['generated_text']

async def run_pipeline(chats):
    print("🧩 Building conversation text...")
    conversation_text = "\n".join([f"{c['user']}: {c['message']}" for c in chats])
    print("🗣️ Conversation:\n", conversation_text)

    print("\n⚡ Running summarization and action extraction in parallel...")
    summary, actions = await asyncio.gather(
        summarize(conversation_text),
        extract_actions(conversation_text)
    )

    return summary, actions

# ==============================================================================
if __name__ == "__main__":
    sample_chats = [
        {"timestamp": "2025-09-22T09:00:00", "user": "Alice", "message": "We need to finalize the budget report by Friday."},
        {"timestamp": "2025-09-22T09:10:00", "user": "Bob", "message": "I'll collect the sales data by Thursday."},
        {"timestamp": "2025-09-22T09:15:00", "user": "Alice", "message": "Also, schedule the team meeting for next week."}
    ]

    cleaned = preprocess_chats(sample_chats)
    summary, actions = asyncio.run(run_pipeline(cleaned))

    print("\n Final Summary:\n", summary)
    print("\n Final Action Items:\n", actions)
