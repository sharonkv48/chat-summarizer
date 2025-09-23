import streamlit as st
import pandas as pd
from datetime import datetime
from collections import Counter
from summarizer import preprocess_chats, run_pipeline
import asyncio

# Helper to parse uploaded CSV
def parse_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    chats = []
    for _, row in df.iterrows():
        chats.append({
            "timestamp": row.get("timestamp", datetime.now().isoformat()),
            "user": row.get("user", "Unknown"),
            "message": row.get("message", "")
        })
    return chats

# Sidebar: Upload and filter
st.sidebar.title("Chat Data Input")
uploaded_file = st.sidebar.file_uploader("Upload chat CSV", type=["csv"])
selected_date = st.sidebar.date_input("Filter by date", value=datetime(2025, 9, 22))

# Load data
if uploaded_file:
    raw_chats = parse_uploaded_file(uploaded_file)
else:
    raw_chats = [
        {"timestamp": "2025-09-22T09:00:00", "user": "Alice", "message": "We need to finalize the budget report by Friday."},
        {"timestamp": "2025-09-22T09:10:00", "user": "Bob", "message": "I'll collect the sales data by Thursday."},
        {"timestamp": "2025-09-22T09:15:00", "user": "Alice", "message": "Also, schedule the team meeting for next week."}
    ]

cleaned_chats = preprocess_chats(raw_chats)
df = pd.DataFrame(cleaned_chats)

# Tabs
tab1, tab2 = st.tabs(["Summary", "Analytics"])

# === Summary Tab ===
with tab1:
    st.header("Conversation Summary")
    full_summary, full_actions = asyncio.run(run_pipeline(cleaned_chats))
    st.subheader("Overall Summary")
    st.write(full_summary)

    st.subheader("Extracted Action Items")
    st.write(full_actions)

# === Analytics Tab ===
with tab2:
    st.header("Conversation Analytics")

    # Filter by date
    filtered_df = df[df["timestamp"].dt.date == selected_date]
    if not filtered_df.empty:
        day_summary, _ = asyncio.run(run_pipeline(filtered_df.to_dict(orient="records")))
        st.subheader(f"Summary for {selected_date}")
        st.write(day_summary)
    else:
        st.info("No messages found for the selected date.")

    # Most common word
    # all_text = " ".join(df["message"].tolist()).lower()
    # words = [word.strip(".,") for word in all_text.split()]
    # common_word = Counter(words).most_common(1)[0][0]
    
    # Most common word (excluding stopwords and punctuation)
    st.subheader("Most Common Word")
    import nltk
    #nltk.download('stopwords')

    from nltk.corpus import stopwords
    import string

    stop_words = set(stopwords.words("english"))
    all_text = " ".join(df["message"].tolist()).lower()

    # Tokenize and clean
    words = [
        word.strip(string.punctuation)
        for word in all_text.split()
        if word.strip(string.punctuation) not in stop_words and word.isalpha()
    ]

    if words:
        common_word = Counter(words).most_common(1)[0][0]
        st.write(common_word)
    else:
        st.write("No meaningful words found.")


    # User activity
    user_counts = df["user"].value_counts()
    st.subheader("User Activity")
    st.write(f"Most Active User: {user_counts.idxmax()} ({user_counts.max()} messages)")
    st.write(f"Least Active User: {user_counts.idxmin()} ({user_counts.min()} messages)")

    # Actionable items priority (basic keyword logic)
    priority_keywords = {"high": ["finalize", "urgent", "asap"], "low": ["schedule", "review", "consider"]}
    action_lines = full_actions.split("\n")
    high_priority = next((a for a in action_lines if any(k in a.lower() for k in priority_keywords["high"])), "None found")
    low_priority = next((a for a in action_lines if any(k in a.lower() for k in priority_keywords["low"])), "None found")

    st.subheader("Priority Actionables")
    st.write(f"Highest Priority Item: {high_priority}")
    st.write(f"Lowest Priority Item: {low_priority}")
