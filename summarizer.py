from transformers import pipeline
from datetime import datetime

summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    use_auth_token="---"  # public alternative to facebook/bart-large-cnn
)

action_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    use_auth_token="---"  # already public
)

def preprocess_chats(chats):
    """Clean messages and normalize timestamps"""
    cleaned = []
    for c in chats:
        if c.get("message") and c['message'].strip():
            try:
                ts = datetime.fromisoformat(c['timestamp'])
            except:
                ts = datetime.now()
            cleaned.append({
                "timestamp": ts,
                "user": c.get("user", "Unknown"),
                "message": c['message'].strip()
            })
    return cleaned

def generate_summary(chats):
    all_messages = " ".join([c['message'] for c in chats])
    summary = summarizer_pipeline(all_messages, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    return summary

def extract_action_items(chats):
    all_messages = " ".join([c['message'] for c in chats])
    prompt = "Extract action items from the following conversation:\n\n" + all_messages
    actions = action_pipeline(prompt, max_length=100)[0]['generated_text']
    return actions

if __name__ == "__main__":
    # Sample chat data
    sample_chats = [
        {"timestamp": "2025-09-22T09:00:00", "user": "Alice", "message": "We need to finalize the budget report by Friday."},
        {"timestamp": "2025-09-22T09:10:00", "user": "Bob", "message": "I'll collect the sales data by Thursday."},
        {"timestamp": "2025-09-22T09:15:00", "user": "Alice", "message": "Also, schedule the team meeting for next week."}
    ]

    # Preprocess
    cleaned_chats = preprocess_chats(sample_chats)
    print("✅ Preprocessed Chats:")
    for c in cleaned_chats:
        print(f"{c['timestamp']} - {c['user']}: {c['message']}")

    # Generate summary
    summary = generate_summary(cleaned_chats)
    print("\n📝 Summary:")
    print(summary)

    # Extract action items
    actions = extract_action_items(cleaned_chats)
    print("\n✅ Action Items:")
    print(actions)