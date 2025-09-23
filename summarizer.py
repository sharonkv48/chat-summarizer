from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==============================================================================

print("Initializing HuggingFace pipelines...")

try:
    summarizer_pipeline = pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum"
    )
    print("Summarizer pipeline loaded successfully")
except Exception as e:
    print("Error loading summarizer pipeline:", e)
    raise

try:
    action_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    print("Action pipeline loaded successfully")
except Exception as e:
    print("Error loading action pipeline:", e)
    raise

print("Wrapping pipelines with LangChain...")
summarizer_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
action_llm = HuggingFacePipeline(pipeline=action_pipeline)

# ========================================================================================

summary_template = PromptTemplate(
    input_variables=["conversation"],
    template="Summarize the following team conversation:\n\n{conversation}\n\nSummary:"
)

action_template = PromptTemplate(
    input_variables=["conversation"],
    template="Extract all action items with deadlines and owners from this conversation:\n\n{conversation}\n\nAction Items:"
)

summary_chain = LLMChain(llm=summarizer_llm, prompt=summary_template)
action_chain = LLMChain(llm=action_llm, prompt=action_template)

# =========================================================================================

def preprocess_chats(chats):
    print(" Preprocessing chats...")
    cleaned = []
    for c in chats:
        if c.get("message") and c['message'].strip():
            try:
                ts = datetime.fromisoformat(c['timestamp'])
            except Exception as e:
                print(f"Invalid timestamp {c['timestamp']}, using now() instead")
                ts = datetime.now()
            cleaned.append({
                "timestamp": ts,
                "user": c.get("user", "Unknown"),
                "message": c['message'].strip()
            })
    print(f"Preprocessed {len(cleaned)} messages")
    return cleaned

def run_pipeline(chats):
    print("Building conversation text...")
    conversation_text = "\n".join([f"{c['user']}: {c['message']}" for c in chats])
    print("Conversation:\n", conversation_text)

    print("\n🚀 Running summarization...")
    summary = summary_chain.run(conversation=conversation_text)
    print("Summary generated")

    print("\n🚀 Extracting action items...")
    actions = action_chain.run(conversation=conversation_text)
    print("Actions extracted")

    return summary, actions

# =============================================================================================

sample_chats = [
    {"timestamp": "2025-09-22T09:00:00", "user": "Alice", "message": "We need to finalize the budget report by Friday."},
    {"timestamp": "2025-09-22T09:10:00", "user": "Bob", "message": "I'll collect the sales data by Thursday."},
    {"timestamp": "2025-09-22T09:15:00", "user": "Alice", "message": "Also, schedule the team meeting for next week."}
]

cleaned = preprocess_chats(sample_chats)
summary, actions = run_pipeline(cleaned)

print("\nFinal Summary:\n", summary)
print("\nFinal Action Items:\n", actions)
