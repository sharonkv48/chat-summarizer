# from google.colab import drive

# # Mount Google Drive
# drive.mount('/content/drive')

from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer


import os, json, nltk, time, re, textwrap

# Add this at the top after imports
try:
    from nltk.tokenize import sent_tokenize
    sent_tokenize("Test.")  # Test if it works
except:
    import nltk
    nltk.download("punkt_tab")
    from nltk.tokenize import sent_tokenize

from nltk.tokenize import sent_tokenize
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

# ========================================================================================

print("Wrapping pipelines with LangChain...")
summarizer_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
action_llm = HuggingFacePipeline(pipeline=action_pipeline)

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
    print("Preprocessing chats...")
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

MAX_TOKENS = 1000  # model max tokens
tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

def chunk_text_by_tokens(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def run_pipeline(chats):
    print("Building conversation text...")
    conversation_text = "\n".join([f"{c['user']}: {c['message']}" for c in chats])
    print(f"Total characters in conversation: {len(conversation_text)}")

    # split into token-based chunks
    chunks = chunk_text_by_tokens(conversation_text)
    print(f"Split conversation into {len(chunks)} chunks")

    summaries = []
    actions_list = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n Processing chunk {i}/{len(chunks)}...")

        # summarization
        summary = summary_chain.invoke({"conversation": chunk})
        summaries.append(summary)

        # action extraction
        actions = action_chain.invoke({"conversation": chunk})
        actions_list.append(actions)

    # Combine all chunk results
    final_summary = " ".join([s['text'] for s in summaries])
    final_actions = " ".join([a['text'] for a in actions_list])

    return final_summary, final_actions

def load_chats_from_json(file_path):
    print(f"Loading chats from {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chat entries")
    return data

# =============================================================================================

    # Testing and Timing Code
# json_file = "preprocessed_data\combined_max_500.json"

# times = []
# for _ in range(1):
#     start = time.perf_counter()
#     chats = load_chats_from_json(json_file)
#     cleaned_chats = preprocess_chats(chats)
#     summary, actions = run_pipeline(cleaned_chats)
#     end = time.perf_counter()
#     elapsed = end - start
#     times.append(elapsed)

# average_time = sum(times) / len(times)
# print(f"Times for each run: {times}")
# print(f"Average time : {average_time:.6f} seconds")

# print("\nFinal Summary:\n", summary)
# print("\nFinal Action Items:\n", actions)
