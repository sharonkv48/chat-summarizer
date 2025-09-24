# For mounting Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import os, json, nltk, time, re, textwrap

# This commnted out section needs to be run once
# nltk.download("punkt")
# nltk.download("punktab")

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
tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

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

def load_chats_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chat entries")
    return data

def preprocess_chats(chats):
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

def chunk_text_by_sentences(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_length = len(sent_tokens)

        if current_length + sent_length > max_tokens and current_chunk:
            chunk_text = tokenizer.decode(sum(current_chunk, []))
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0

        current_chunk.append(sent_tokens)
        current_length += sent_length

    if current_chunk:
        chunk_text = tokenizer.decode(sum(current_chunk, []))
        chunks.append(chunk_text)

    return chunks

def run_pipeline(chats):
    conversation_text = "\n".join([f"{c['user']}: {c['message']}" for c in chats])
    print(f"Total characters in conversation: {len(conversation_text)}")

    chunks = chunk_text_by_sentences(conversation_text)
    print(f"Split conversation into {len(chunks)} chunks")

    summaries = []
    actions_list = []

    for i, chunk in enumerate(chunks, 1):
        print(f"\n Processing chunk {i}/{len(chunks)}...")

        summary = summary_chain.invoke({"conversation": chunk})
        summaries.append(summary)

    final_summary = " ".join([s['text'] for s in summaries])
    final_actions = action_chain.invoke({"conversation": final_summary})

    final_summary = clean_summary(final_summary)
    final_actions_clean = format_action_items(final_actions['text'])

    pretty_print("Final Summary", final_summary)
    pretty_print("Final Action Items", final_actions_clean)

    return final_summary, final_actions_clean

def clean_summary(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    seen = set()
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            cleaned_sentences.append(s)

    return " ".join(cleaned_sentences)
    
def format_action_items(text):
    items = re.split(r'[\n;,.]', text)
    seen, bullets = set(), []
    for item in items:
        item = item.strip("-• ").strip()
        if item and item not in seen:
            seen.add(item)
            bullets.append(f"- {item}")
    return "\n".join(bullets)

def pretty_print(title, text, width=100):
    print(f"\n{title}:")
    print("=" * len(title))
    print(textwrap.fill(text, width=width))

# =============================================================================================

    # Testing and Timing Code
# json_file = "/content/drive/MyDrive/combined_max_500.json"

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

