# For mounting Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import os, json, nltk, time, re, textwrap

# This commented out section needs to be run once
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import json 

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ==============================================================================

print("Initializing HuggingFace pipelines...")

# Enhanced model configuration with multiple options
MODEL_CONFIGS = {
    "fast": {
        "summarizer": "sshleifer/distilbart-cnn-12-6",
        "action": "google/flan-t5-small",
        "summarizer_max_length": 100,
        "summarizer_min_length": 30,
        "action_max_length": 150
    },
    "balanced": {
        "summarizer": "philschmid/bart-large-cnn-samsum",
        "action": "google/flan-t5-base", 
        "summarizer_max_length": 150,
        "summarizer_min_length": 40,
        "action_max_length": 200
    },
    "quality": {
        "summarizer": "facebook/bart-large-cnn",
        "action": "google/flan-t5-large",
        "summarizer_max_length": 200,
        "summarizer_min_length": 50,
        "action_max_length": 250
    }
}

def initialize_pipelines(model_choice="balanced"):
    """Initialize HuggingFace pipelines with model choice"""
    config = MODEL_CONFIGS.get(model_choice, MODEL_CONFIGS["balanced"])
    
    try:
        summarizer_pipeline = pipeline(
            "summarization",
            model=config["summarizer"],
            max_length=config["summarizer_max_length"],
            min_length=config["summarizer_min_length"],
            truncation=True
        )
        print(f"Summarizer pipeline ({config['summarizer']}) loaded successfully")
    except Exception as e:
        print("Error loading summarizer pipeline:", e)
        return None, None

    try:
        action_pipeline = pipeline(
            "text2text-generation",
            model=config["action"],
            max_length=config["action_max_length"],
            truncation=True
        )
        print(f"Action pipeline ({config['action']}) loaded successfully")
    except Exception as e:
        print("Error loading action pipeline:", e)
        return None, None

    return summarizer_pipeline, action_pipeline

# ========================================================================================

print("Wrapping pipelines with LangChain...")

# Initialize with balanced model by default
summarizer_pipeline, action_pipeline = initialize_pipelines("balanced")

if summarizer_pipeline and action_pipeline:
    summarizer_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
    action_llm = HuggingFacePipeline(pipeline=action_pipeline)
    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
else:
    # Fallback to mock mode
    summarizer_llm = None
    action_llm = None
    tokenizer = None

summary_template = PromptTemplate(
    input_variables=["conversation"],
    template="Summarize the following team conversation:\n\n{conversation}\n\nSummary:"
)

action_template = PromptTemplate(
    input_variables=["conversation"],
    template="Extract all action items with deadlines and owners from this conversation:\n\n{conversation}\n\nAction Items:"
)

if summarizer_llm and action_llm:
    summary_chain = LLMChain(llm=summarizer_llm, prompt=summary_template)
    action_chain = LLMChain(llm=action_llm, prompt=action_template)
else:
    summary_chain = None
    action_chain = None

# =========================================================================================

def load_chats_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} chat entries")
    return data

def preprocess_chats(chats):
    """Clean and preprocess chat data"""
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

def chunk_text_by_sentences(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False) if tokenizer else []
        sent_length = len(sent_tokens) if sent_tokens else len(sent) // 4  # rough estimate

        if current_length + sent_length > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [sent]
            current_length = sent_length
        else:
            current_chunk.append(sent)
            current_length += sent_length

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)

    return chunks

def run_pipeline(chats, model_choice="balanced", process_all_chunks=False):
    """Enhanced run_pipeline with model choice and chunk processing control"""
    
    # Reinitialize pipelines if model choice changed
    global summarizer_pipeline, action_pipeline, summarizer_llm, action_llm, summary_chain, action_chain
    
    if model_choice != "balanced":  # Only reinitialize if not using default
        summarizer_pipeline, action_pipeline = initialize_pipelines(model_choice)
        if summarizer_pipeline and action_pipeline:
            summarizer_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
            action_llm = HuggingFacePipeline(pipeline=action_pipeline)
            summary_chain = LLMChain(llm=summarizer_llm, prompt=summary_template)
            action_chain = LLMChain(llm=action_llm, prompt=action_template)
    
    # Fallback to mock mode if pipelines not available
    if not summary_chain or not action_chain:
        return generate_mock_summary(chats)
    
    conversation_text = "\n".join([f"{c['user']}: {c['message']}" for c in chats])
    print(f"Total characters in conversation: {len(conversation_text)}")
    print(f"Using model: {model_choice}")

    chunks = chunk_text_by_sentences(conversation_text)
    print(f"Split conversation into {len(chunks)} chunks")

    summaries = []
    actions_list = []

    # Determine how many chunks to process
    if process_all_chunks:
        chunks_to_process = chunks
    else:
        # For demo purposes, limit to first 2 chunks to avoid timeout
        chunks_to_process = chunks[:2] if len(chunks) > 1 else chunks[:1]
        if len(chunks) > 2:
            print(f"Processing first {len(chunks_to_process)} chunks for demo (use process_all_chunks=True for full processing)")

    for i, chunk in enumerate(chunks_to_process, 1):
        print(f"Processing chunk {i}/{len(chunks_to_process)}...")

        try:
            summary = summary_chain.invoke({"conversation": chunk})
            summaries.append(summary['text'] if isinstance(summary, dict) else summary)
            
            # Use summary for action extraction to reduce processing time
            action = action_chain.invoke({"conversation": chunk})
            actions_list.append(action['text'] if isinstance(action, dict) else action)
            
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            # Add mock content for failed chunks
            summaries.append(f"Summary for chunk {i} unavailable.")
            actions_list.append(f"Actions for chunk {i} unavailable.")

    final_summary = " ".join(summaries)
    final_actions = " ".join(actions_list)

    final_summary = clean_summary(final_summary)
    final_actions_clean = format_action_items(final_actions)

    pretty_print("Final Summary", final_summary)
    pretty_print("Final Action Items", final_actions_clean)

    return final_summary, final_actions_clean

def generate_mock_summary(chats):
    """Generate mock summary when AI models are not available"""
    total_messages = len(chats)
    unique_users = len(set(ch['user'] for ch in chats))
    
    summary = f"""
    📊 Conversation Summary (Mock Mode)
    
    **Overview:** Analyzed {total_messages} messages from {unique_users} participants.
    
    **Key Discussion Points:**
    - Project timelines and deadline management
    - Technical implementation details
    - Team coordination and resource allocation
    - Client requirements and feedback integration
    
    **Sentiment Analysis:** Mostly positive and collaborative
    
    *Enable AI features for detailed analysis by installing:*
    `pip install torch transformers langchain langchain-community`
    """
    
    actions = """
    ✅ **Action Items:**
    
    - Review project implementation timeline (Due: This week)
    - Schedule team coordination meeting (Owner: Team Lead)
    - Update documentation with latest decisions (Due: Next meeting)
    - Assign tasks for upcoming deliverables (Owner: Project Manager)
    
    *AI-powered action extraction available with package installation.*
    """
    
    return summary, actions

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
# Enhanced ChatSummarizer class for Streamlit integration
# =============================================================================================

class ChatSummarizer:
    def __init__(self, use_mock=False, model_choice="balanced"):
        self.model_choice = model_choice
        self.use_mock = use_mock or (summary_chain is None and action_chain is None)
        
        if self.use_mock:
            print("Running in mock mode")
        else:
            print(f"Initialized with model: {model_choice}")
    
    def summarize_conversation(self, chats, process_all_chunks=False):
        """Main method to summarize conversations"""
        if self.use_mock:
            return generate_mock_summary(chats)
        
        return run_pipeline(chats, self.model_choice, process_all_chunks)
    
    def run_pipeline(self, chats, process_all_chunks=False):
        """Alias for summarize_conversation for compatibility"""
        return self.summarize_conversation(chats, process_all_chunks)

# =============================================================================================
# Backward compatibility functions
# =============================================================================================

# For direct function calls (maintains old interface)
def run_pipeline_legacy(chats):
    """Legacy function for backward compatibility"""
    return run_pipeline(chats, "balanced", False)

# Alias for easy import
def summarize_chats(chats, model_choice="balanced", process_all_chunks=False):
    """Convenience function for direct summarization"""
    summarizer = ChatSummarizer(model_choice=model_choice)
    return summarizer.summarize_conversation(chats, process_all_chunks)

# =============================================================================================
# Testing and Timing Code
# =============================================================================================

def test_performance(json_file_path, num_runs=1, model_choice="balanced"):
    """Test performance with timing"""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        chats = load_chats_from_json(json_file_path)
        cleaned_chats = preprocess_chats(chats)
        summary, actions = run_pipeline(cleaned_chats, model_choice)
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)

    average_time = sum(times) / len(times)
    print(f"Model: {model_choice}")
    print(f"Times for each run: {times}")
    print(f"Average time: {average_time:.2f} seconds")
    return average_time

if __name__ == "__main__":
    # Example usage with different models
    sample_chats = [
        {
            "timestamp": "2024-01-15T10:30:00",
            "user": "Alice",
            "message": "Let's discuss the Q1 project timeline. We need to finalize the deliverables."
        },
        {
            "timestamp": "2024-01-15T10:35:00", 
            "user": "Bob",
            "message": "I agree. We should schedule a meeting with the development team."
        },
        {
            "timestamp": "2024-01-15T10:40:00",
            "user": "Charlie", 
            "message": "I'll prepare the agenda and send it out by tomorrow morning."
        }
    ]
    
    
    print("Testing different model configurations:\n")
    
    # Test fast model
    print("="*50)
    print("FAST MODEL RESULTS:")
    summarizer_fast = ChatSummarizer(model_choice="fast")
    summary, actions = summarizer_fast.summarize_conversation(sample_chats)
    
    print("\n" + "="*50)
    print("BALANCED MODEL RESULTS:")
    summarizer_balanced = ChatSummarizer(model_choice="balanced")
    summary, actions = summarizer_balanced.summarize_conversation(sample_chats)
    
    print("\n" + "="*50)
    print("QUALITY MODEL RESULTS:")
    summarizer_quality = ChatSummarizer(model_choice="quality")
    summary, actions = summarizer_quality.summarize_conversation(sample_chats)