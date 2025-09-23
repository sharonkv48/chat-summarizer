import streamlit as st
import pandas as pd
from datetime import datetime
from collections import Counter
from summarizer import preprocess_chats, run_pipeline
import json
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# === Helper to parse uploaded files ===
def parse_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    chats = []
    for _, row in df.iterrows():
        chats.append({
            "timestamp": row.get("timestamp", datetime.now().isoformat()),
            "user": row.get("user", "Unknown"),
            "message": row.get("message", "")
        })
    return chats

def parse_json(uploaded_file):
    try:
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return []

# === Sidebar UI ===
st.sidebar.title("📥 Chat Data Input")

data_source = st.sidebar.radio("Choose data source:", ["Upload File", "Use Sample Data", "Use Preprocessed File"])
uploaded_file = None
raw_chats = []

if data_source == "Upload File":
    file_type = st.sidebar.selectbox("File type:", ["CSV", "JSON"])
    uploaded_file = st.sidebar.file_uploader("Upload your chat file", type=["csv", "json"])
    if uploaded_file:
        raw_chats = parse_csv(uploaded_file) if file_type == "CSV" else parse_json(uploaded_file)
else:
    if data_source == "Use Sample Data":
        raw_chats = [
            {"timestamp": "2025-09-22T09:00:00", "user": "Alice", "message": "We need to finalize the budget report by Friday."},
            {"timestamp": "2025-09-22T09:10:00", "user": "Bob", "message": "I'll collect the sales data by Thursday."},
            {"timestamp": "2025-09-22T09:15:00", "user": "Alice", "message": "Also, schedule the team meeting for next week."}
        ]
    else:
        file_path = "preprocessed_data/combined_max_500.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_chats = json.load(f)
            st.sidebar.success(f"Loaded {len(raw_chats)} chat entries from file.")
        except Exception as e:
            st.sidebar.error(f"Failed to load file: {e}")

# === Early exit if no data ===
if not raw_chats:
    st.warning("No data loaded. Please select a data source and upload a file if needed.")
    st.stop()

# === Preprocess and Filter ===
cleaned_chats = preprocess_chats(raw_chats)
df = pd.DataFrame(cleaned_chats)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Display total loaded data info
st.sidebar.write(f"**Total messages loaded:** {len(df)}")

# === Timestamp Range Selection ===
min_ts = df["timestamp"].min().to_pydatetime()
max_ts = df["timestamp"].max().to_pydatetime()

start_ts, end_ts = st.sidebar.slider(
    "🕒 Select time range",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts),
    format="YYYY-MM-DD HH:mm"
)

# Filter data based on timestamp range
filtered_df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

# Display filtered data info
st.sidebar.write(f"**Messages in selected range:** {len(filtered_df)}")

if len(filtered_df) == 0:
    st.warning("No messages found in the selected time range. Please adjust the time range.")
    st.stop()

# === Preview Uploaded Data ===
with st.expander("📂 Preview Filtered Data", expanded=False):
    st.write(f"**Showing {len(filtered_df)} messages from selected time range:**")
    st.dataframe(filtered_df, use_container_width=True)

# === Tabs ===
tab1, tab2 = st.tabs(["📝 Summary", "📊 Analytics"])

# === Summary Tab ===
with tab1:
    st.header("📝 Conversation Summary")
    
    # Convert filtered dataframe back to chat format for the summarizer
    filtered_chats = []
    for _, row in filtered_df.iterrows():
        filtered_chats.append({
            "timestamp": row["timestamp"].isoformat(),
            "user": row["user"],
            "message": row["message"]
        })
    
    # Show what will be processed
    conversation_preview = "\n".join([f"{c['user']}: {c['message']}" for c in filtered_chats[:5]])
    total_chars = sum(len(c['message']) for c in filtered_chats)
    
    st.info(f"""
    **About to process:**
    - {len(filtered_chats)} messages
    - {total_chars:,} total characters
    - Time range: {start_ts} to {end_ts}
    """)
    
    with st.expander("📋 Preview of conversation to be summarized"):
        st.text(conversation_preview + "\n..." if len(filtered_chats) > 5 else conversation_preview)

    if st.button("Generate Summary & Action Items", type="primary"):
        with st.spinner(f"Processing {len(filtered_chats)} messages..."):
            try:
                # Call your existing run_pipeline function with filtered data only
                full_summary, full_actions = run_pipeline(filtered_chats)

                st.subheader("🔍 Summary")
                st.write(full_summary)

                st.subheader("✅ Action Items")
                st.write(full_actions)
                
                # Store in session state for analytics tab
                st.session_state['last_summary'] = full_summary
                st.session_state['last_actions'] = full_actions
                
            except Exception as e:
                st.error(f"Error generating summary: {e}")

# === Analytics Tab ===
with tab2:
    st.header("📊 Conversation Analytics")
    
    # Convert filtered dataframe back to chat format
    filtered_chats = []
    for _, row in filtered_df.iterrows():
        filtered_chats.append({
            "timestamp": row["timestamp"].isoformat(),
            "user": row["user"],
            "message": row["message"]
        })
    
    st.info(f"Analyzing {len(filtered_chats)} messages from selected time range")

    # Generate summary for analytics if not already done
    if 'last_summary' not in st.session_state:
        if st.button("Generate Analytics Summary", type="secondary"):
            with st.spinner(f"Analyzing {len(filtered_chats)} messages..."):
                try:
                    day_summary, day_actions = run_pipeline(filtered_chats)
                    st.session_state['last_summary'] = day_summary
                    st.session_state['last_actions'] = day_actions
                except Exception as e:
                    st.error(f"Error generating analytics: {e}")
                    st.stop()
    
    if 'last_summary' in st.session_state:
        st.subheader("🗓️ Summary for Selected Range")
        st.write(st.session_state['last_summary'])

    # Most common word
    st.subheader("🔠 Most Common Word")
    all_text = " ".join(filtered_df["message"].tolist()).lower()
    words = [
        word.strip(string.punctuation)
        for word in all_text.split()
        if word.strip(string.punctuation) not in stop_words and word.isalpha()
    ]
    if words:
        word_counts = Counter(words)
        common_word, count = word_counts.most_common(1)[0]
        st.write(f"**{common_word}** (appeared {count} times)")
        
        # Show top 10 words
        with st.expander("Top 10 most common words"):
            top_words = word_counts.most_common(10)
            for word, count in top_words:
                st.write(f"- {word}: {count}")
    else:
        st.write("No meaningful words found.")

    # User activity
    st.subheader("👥 Messages Per User")
    user_counts = filtered_df["user"].value_counts()
    st.bar_chart(user_counts)
    
    # Show user stats
    total_users = len(user_counts)
    most_active = user_counts.index[0]
    st.write(f"**{total_users}** users participated. **{most_active}** was most active with **{user_counts[most_active]}** messages.")

    # Message frequency over time
    st.subheader("📈 Message Frequency Over Time")
    if len(filtered_df) > 1:
        time_series = filtered_df.groupby(filtered_df["timestamp"].dt.floor("H")).size()
        st.line_chart(time_series)
        
        # Peak activity
        peak_hour = time_series.idxmax()
        peak_count = time_series.max()
        st.write(f"**Peak activity:** {peak_count} messages at {peak_hour.strftime('%Y-%m-%d %H:00')}")
    else:
        st.write("Not enough data points for time series analysis")

    # Hourly activity heatmap
    st.subheader("🌡️ Hourly Activity Heatmap")
    if len(filtered_df) > 5:  # Only show if we have enough data
        heatmap_data = filtered_df.copy()
        heatmap_data["hour"] = heatmap_data["timestamp"].dt.hour
        heatmap_data["day"] = heatmap_data["timestamp"].dt.date
        pivot_table = heatmap_data.pivot_table(index="day", columns="hour", aggfunc="size", fill_value=0)

        if not pivot_table.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_table, cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Messages'})
            ax.set_title("Messages by Hour and Day")
            st.pyplot(fig)
        else:
            st.write("Not enough data for heatmap")
    else:
        st.write("Not enough data for heatmap (need more than 5 messages)")

    # Priority actionables (only if we have actions)
    if 'last_actions' in st.session_state:
        st.subheader("⚡ Priority Actionables")
        priority_keywords = {
            "high": ["finalize", "urgent", "asap", "immediately", "critical", "priority"],
            "medium": ["complete", "finish", "deliver", "submit"],
            "low": ["schedule", "review", "consider", "think about", "maybe"]
        }
        
        action_lines = [line.strip() for line in st.session_state['last_actions'].split("\n") if line.strip()]
        
        high_priority = [a for a in action_lines if any(k in a.lower() for k in priority_keywords["high"])]
        medium_priority = [a for a in action_lines if any(k in a.lower() for k in priority_keywords["medium"])]
        low_priority = [a for a in action_lines if any(k in a.lower() for k in priority_keywords["low"])]

        if high_priority:
            st.write("🔴 **High Priority:**")
            for item in high_priority[:3]:  # Show top 3
                st.write(f"- {item}")
        
        if medium_priority:
            st.write("🟡 **Medium Priority:**")
            for item in medium_priority[:3]:
                st.write(f"- {item}")
                
        if low_priority:
            st.write("🟢 **Low Priority:**")
            for item in low_priority[:3]:
                st.write(f"- {item}")
        
        if not (high_priority or medium_priority or low_priority):
            st.write("No priority indicators found in action items")

# === Footer ===
st.markdown("---")
st.markdown(f"*Processed {len(filtered_df)} messages from {start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')}*")
