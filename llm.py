from transformers import pipeline

# Sample LLM
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

summary = summarizer(conversation_text, max_length=60, min_length=10, do_sample=False)[0]['summary_text']

print("Chat Summary:", summary)
