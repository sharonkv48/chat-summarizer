import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
import json
import sys
import os
from fpdf import FPDF

# Add the current directory to path to import summarizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from summarizer import ChatSummarizer, preprocess_chats, run_pipeline, summarize_chats
    SUMMARIZER_AVAILABLE = True
except ImportError as e:
    SUMMARIZER_AVAILABLE = False
    print(f"Summarizer import warning: {e}")

# Configure the page
st.set_page_config(
    page_title="Enterprise Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'chat_collapsed' not in st.session_state:
    st.session_state.chat_collapsed = False
if 'flagged_items' not in st.session_state:
    st.session_state.flagged_items = []
if 'star_ratings' not in st.session_state:
    st.session_state.star_ratings = {}
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = ""
if 'last_actions' not in st.session_state:
    st.session_state.last_actions = ""
if 'current_view' not in st.session_state:
    st.session_state.current_view = "Chat Analysis"
if 'show_summary' not in st.session_state:
    st.session_state.show_summary = False
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "balanced"
if 'process_all_chunks' not in st.session_state:
    st.session_state.process_all_chunks = False
if 'summarizer_initialized' not in st.session_state:
    st.session_state.summarizer_initialized = False

# Custom CSS for Light Professional Theme
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #f59e0b;
        --surface-color: #ffffff;
        --text-color: #374151;
        --card-color: #f9fafb;
        --border-color: #e5e7eb;
        --bg-color: #f8fafc;
    }

    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }

    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        color: var(--primary);
        border-bottom: 3px solid var(--primary);
        padding-bottom: 1rem;
    }

    .summary-section {
        background: var(--surface-color);
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .summary-box {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid var(--primary);
    }

    .action-box {
        background: var(--card-color);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid var(--secondary);
    }

    .chat-container {
        background: var(--surface-color);
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
        overflow: hidden;
    }

    .chat-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .chat-body {
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
    }

    .chat-body.collapsed {
        max-height: 0;
        padding: 0;
        overflow: hidden;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 6px;
        max-width: 70%;
        word-wrap: break-word;
    }

    .user-message {
        background: var(--primary);
        color: white;
        margin-left: auto;
    }

    .other-message {
        background: var(--card-color);
        color: var(--text-color);
        margin-right: auto;
        border: 1px solid var(--border-color);
    }

    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
    }

    .message-time {
        font-size: 0.75rem;
        opacity: 0.8;
    }

    .model-option {
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
        cursor: pointer;
        background: white;
    }

    .model-option.selected {
        border-color: var(--primary);
        background: rgba(37, 99, 235, 0.05);
        transform: scale(1.02);
    }

    .model-option:hover {
        border-color: var(--primary);
        transform: translateY(-1px);
    }

    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    .status-fast { background: #10b981; color: white; }
    .status-balanced { background: #3b82f6; color: white; }
    .status-quality { background: #8b5cf6; color: white; }
    .status-mock { background: #6b7280; color: white; }

    .processing-info {
        background: var(--card-color);
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid var(--secondary);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .insight-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

class PDFGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
    
    def generate_pdf(self, title, content, flagged_items):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, title, 0, 1, 'C')
        self.pdf.ln(10)
        
        self.pdf.set_font("Arial", size=12)
        
        def clean_text(text):
            return text.encode('latin-1', 'replace').decode('latin-1')
        
        for line in content.split('\n'):
            if line.strip():
                self.pdf.multi_cell(0, 10, clean_text(line.strip()))
                self.pdf.ln(5)
        
        if flagged_items:
            self.pdf.ln(10)
            self.pdf.set_font("Arial", 'B', 14)
            self.pdf.cell(0, 10, "Important Points:", 0, 1)
            self.pdf.ln(5)
            
            self.pdf.set_font("Arial", size=11)
            for item in flagged_items:
                self.pdf.multi_cell(0, 8, f"- {clean_text(item)}")
                self.pdf.ln(3)
        
        return self.pdf.output(dest='S').encode('latin-1')

class EnhancedChatApp:
    def __init__(self):
        self.raw_chats = []
        self.df = pd.DataFrame()
        self.initialize_summarizer()
    
    def initialize_summarizer(self):
        """Initialize summarizer with current settings"""
        if not SUMMARIZER_AVAILABLE:
            st.session_state.summarizer = None
            return
        
        # Only reinitialize if settings changed or not initialized
        if (not st.session_state.summarizer_initialized or 
            hasattr(st.session_state, 'current_model') and 
            st.session_state.current_model != st.session_state.model_choice):
            
            try:
                st.session_state.summarizer = ChatSummarizer(
                    use_mock=False,
                    model_choice=st.session_state.model_choice
                )
                st.session_state.current_model = st.session_state.model_choice
                st.session_state.summarizer_initialized = True
            except Exception as e:
                st.error(f"Error initializing summarizer: {e}")
                st.session_state.summarizer = None
    
    def update_model_choice(self, new_choice):
        """Update model choice"""
        st.session_state.model_choice = new_choice
        st.session_state.summarizer_initialized = False
        self.initialize_summarizer()
        st.rerun()
    
    def get_model_display_info(self):
        """Get display information for current model"""
        models = {
            "fast": {
                "name": "Fast",
                "description": "Quick results for large conversations",
                "tech": "DistilBART + Flan-T5-small",
                "status_class": "status-fast"
            },
            "balanced": {
                "name": "Balanced", 
                "description": "Good balance of speed and quality",
                "tech": "BART-large + Flan-T5-base",
                "status_class": "status-balanced"
            },
            "quality": {
                "name": "Quality",
                "description": "Best results for important analysis",
                "tech": "BART-large-cnn + Flan-T5-large",
                "status_class": "status-quality"
            }
        }
        return models.get(st.session_state.model_choice, models["balanced"])
    
    def parse_uploaded_file(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.json'):
                return json.load(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
                return [
                    {
                        "timestamp": row.get("timestamp", datetime.now().isoformat()),
                        "user": row.get("user", "Unknown"),
                        "message": row.get("message", ""),
                    }
                    for _, row in df.iterrows()
                ]
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            return []
    
    def get_chat_metrics(self, chats, selected_date=None):
        """Calculate chat metrics"""
        filtered_chats = [chat for chat in chats 
                         if not selected_date or 
                         pd.to_datetime(chat['timestamp']).date() == selected_date]
        
        if not filtered_chats:
            return {}
        
        total_messages = len(filtered_chats)
        unique_users = len(set(chat['user'] for chat in filtered_chats))
        avg_message_length = sum(len(str(chat['message'])) for chat in filtered_chats) / total_messages
        
        # Extract common topics
        common_words = self.extract_common_topics([chat['message'] for chat in filtered_chats])
        
        # Time-based metrics
        timestamps = [pd.to_datetime(chat['timestamp']) for chat in filtered_chats]
        time_span = max(timestamps) - min(timestamps) if timestamps else timedelta(0)
        
        return {
            'total_messages': total_messages,
            'unique_users': unique_users,
            'avg_message_length': round(avg_message_length, 1),
            'time_span_hours': round(time_span.total_seconds() / 3600, 1),
            'common_topics': common_words[:10],
            'date_range': f"{min(timestamps).strftime('%Y-%m-%d %H:%M')} to {max(timestamps).strftime('%Y-%m-%d %H:%M')}" if timestamps else "N/A"
        }
    
    def extract_common_topics(self, messages):
        """Extract common topics from messages"""
        try:
            try:
                stopwords.words("english")
            except:
                nltk.download('stopwords')
            
            stop_words = set(stopwords.words("english"))
            all_text = " ".join([str(msg) for msg in messages]).lower()
            
            words = [
                word.strip(string.punctuation)
                for word in all_text.split()
                if word.strip(string.punctuation) not in stop_words 
                and len(word.strip(string.punctuation)) > 3
            ]
            
            word_counts = Counter(words).most_common(15)
            return [word for word, count in word_counts]
        except:
            return []
    
    def detect_tickets_raised(self, messages):
        """Detect potential tickets raised in messages"""
        ticket_keywords = ['ticket', 'issue', 'problem', 'bug', 'error', 'request', 'support', 'help', 'fix']
        ticket_messages = []
        
        for msg in messages:
            msg_lower = str(msg).lower()
            if any(keyword in msg_lower for keyword in ticket_keywords):
                ticket_messages.append(msg)
        
        return len(ticket_messages), ticket_messages[:5]  # Return count and sample messages
    
    def display_model_selection(self):
        """Display model selection options"""
        st.markdown("### AI Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        models_config = {
            "fast": {"name": "Fast", "desc": "Quick results", "tech": "DistilBART + Flan-T5-small"},
            "balanced": {"name": "Balanced", "desc": "Best balance", "tech": "BART-large + Flan-T5-base"},
            "quality": {"name": "Quality", "desc": "Best quality", "tech": "BART-large-cnn + Flan-T5-large"}
        }
        
        for model_key, col in zip(["fast", "balanced", "quality"], [col1, col2, col3]):
            with col:
                is_selected = st.session_state.model_choice == model_key
                model_info = models_config[model_key]
                
                st.markdown(f'''
                <div class="model-option {'selected' if is_selected else ''}"
                     onclick="this.closest('div').querySelector('button').click()">
                    <h4>{model_info['name']}</h4>
                    <p>{model_info['desc']}</p>
                    <small>{model_info['tech']}</small>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.button(f"Select {model_info['name']}", key=f"{model_key}_btn", use_container_width=True):
                    self.update_model_choice(model_key)
        
        # Processing options
        st.markdown("---")
        st.checkbox(
            "Process all conversation chunks (for large chats)",
            value=st.session_state.process_all_chunks,
            key="process_all_chunks",
            help="When unchecked, processes only first 2 chunks for faster results"
        )
    
    def display_summary_section(self):
        """Display the generated summary"""
        if st.session_state.show_summary and st.session_state.last_summary:
            model_info = self.get_model_display_info()
            
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            st.markdown("### AI Analysis Results")
            
            # Model info badge
            st.markdown(f'''
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span>Current Model:</span>
                <span class="status-badge {model_info['status_class']}">
                    {model_info['name']} • {model_info['tech']}
                </span>
                <span style="margin-left: auto;">
                    Processing: {'All chunks' if st.session_state.process_all_chunks else 'Demo mode (2 chunks)'}
                </span>
            </div>
            ''', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown("#### Conversation Summary")
                st.write(st.session_state.last_summary)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="action-box">', unsafe_allow_html=True)
                st.markdown("#### Action Items")
                st.write(st.session_state.last_actions)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def display_chat_interface(self, chats, selected_date=None):
        """Main chat analysis interface"""
        filtered_chats = [chat for chat in chats 
                         if not selected_date or 
                         pd.to_datetime(chat['timestamp']).date() == selected_date]
        
        # Model configuration
        self.display_model_selection()
        
        # Summary display
        self.display_summary_section()
        
        # Analysis controls
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("Generate AI Summary", type="primary", use_container_width=True):
                self.generate_summary(filtered_chats)
        
        with col2:
            if st.session_state.show_summary:
                if st.button("Clear", use_container_width=True):
                    self.clear_summary()
        
        with col3:
            if st.button("Toggle Chat", use_container_width=True):
                st.session_state.chat_collapsed = not st.session_state.chat_collapsed
                st.rerun()
        
        # Chat display
        self.display_chat_messages(filtered_chats, selected_date)
        
        # Metrics
        if filtered_chats:
            metrics = self.get_chat_metrics(chats, selected_date)
            self.display_chat_metrics(metrics)
    
    def generate_summary(self, chats):
        """Generate summary using the selected model"""
        if not chats:
            st.warning("No messages to analyze")
            return
        
        with st.spinner("Analyzing conversations with AI..."):
            try:
                if st.session_state.summarizer:
                    # Use the class method
                    summary, actions = st.session_state.summarizer.summarize_conversation(
                        chats, 
                        process_all_chunks=st.session_state.process_all_chunks
                    )
                else:
                    # Fallback to direct function call
                    summary, actions = summarize_chats(
                        chats,
                        model_choice=st.session_state.model_choice,
                        process_all_chunks=st.session_state.process_all_chunks
                    )
                
                st.session_state.last_summary = summary
                st.session_state.last_actions = actions
                st.session_state.show_summary = True
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                # Fallback to mock mode
                if st.session_state.summarizer:
                    summary, actions = st.session_state.summarizer.summarize_conversation(chats)
                else:
                    from summarizer import generate_mock_summary
                    summary, actions = generate_mock_summary(chats)
                
                st.session_state.last_summary = summary
                st.session_state.last_actions = actions
                st.session_state.show_summary = True
                st.info("Using mock summary due to analysis error")
    
    def clear_summary(self):
        """Clear current summary"""
        st.session_state.show_summary = False
        st.session_state.last_summary = ""
        st.session_state.last_actions = ""
        st.rerun()
    
    def display_chat_messages(self, chats, selected_date):
        """Display chat messages"""
        st.markdown(f'''
        <div class="chat-container">
            <div class="chat-header">
                <div>
                    <h3 style="margin:0;">Live Chat Window</h3>
                    <small>{selected_date if selected_date else "All Dates"} • {len(chats)} messages</small>
                </div>
                <div>{'Collapsed' if st.session_state.chat_collapsed else 'Expanded'}</div>
            </div>
            <div class="chat-body {'collapsed' if st.session_state.chat_collapsed else ''}">
        ''', unsafe_allow_html=True)
        
        if not st.session_state.chat_collapsed and chats:
            sorted_chats = sorted(chats, key=lambda x: pd.to_datetime(x['timestamp']))
            
            for i, chat in enumerate(sorted_chats):
                timestamp = pd.to_datetime(chat['timestamp'])
                message_class = "user-message" if i % 2 == 0 else "other-message"
                
                st.markdown(f'''
                <div class="chat-message {message_class}">
                    <div class="message-header">
                        <span>{chat['user']}</span>
                        <span class="message-time">{timestamp.strftime('%H:%M:%S')}</span>
                    </div>
                    <div>{chat['message']}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    def display_chat_metrics(self, metrics):
        """Display chat metrics"""
        st.markdown("### Chat Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics_data = [
            (metrics["total_messages"], "Total Messages"),
            (metrics["unique_users"], "Unique Users"),
            (f"{metrics['avg_message_length']:.0f}", "Avg Length"),
            (f"{metrics['time_span_hours']}h", "Time Span"),
            (len(metrics["common_topics"]), "Key Topics")
        ]
        
        for col, (value, label) in zip([col1, col2, col3, col4, col5], metrics_data):
            with col:
                st.metric(label, value)
        
        if metrics['common_topics']:
            st.write(f"**Common Topics:** {', '.join(metrics['common_topics'])}")
        st.write(f"**Date Range:** {metrics['date_range']}")
    
    def display_documentation_mode(self):
        """Documentation mode with star ratings"""
        if not st.session_state.last_summary:
            st.info("Generate a summary first in Chat Analysis mode")
            return
        
        st.markdown("### Documentation Mode")
        st.markdown("Rate important points with stars for PDF export")
        
        # Simple documentation interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summary")
            st.text_area("Summary", st.session_state.last_summary, height=200, key="doc_summary")
        
        with col2:
            st.subheader("Action Items")
            st.text_area("Actions", st.session_state.last_actions, height=200, key="doc_actions")
        
        # PDF Export
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            try:
                pdf_gen = PDFGenerator()
                pdf_data = pdf_gen.generate_pdf(
                    "Chat Analysis Report",
                    f"Summary:\n{st.session_state.last_summary}\n\nAction Items:\n{st.session_state.last_actions}",
                    []  # Simplified without flagged items
                )
                
                st.download_button(
                    "Download PDF",
                    pdf_data,
                    "chat_analysis_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
    
    def display_analytics_dashboard(self, df, selected_date=None):
        """Enhanced Analytics dashboard with insights"""
        st.markdown("### Analytics Dashboard")
        
        analytics_df = df[df['timestamp'].dt.date == selected_date] if selected_date else df
        
        if analytics_df.empty:
            st.info("No data available for selected date")
            return
        
        # Enhanced Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: 
            st.metric("Total Messages", len(analytics_df))
        with col2: 
            st.metric("Unique Users", analytics_df['user'].nunique())
        with col3: 
            st.metric("Avg Message Length", f"{analytics_df['message'].str.len().mean():.0f} chars")
        with col4:
            tickets_count, _ = self.detect_tickets_raised(analytics_df['message'])
            st.metric("Tickets Raised", tickets_count)
        with col5:
            st.metric("Date", str(selected_date) if selected_date else "All Dates")
        
        # Row 1: Message Frequency and Top Users
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Message Frequency Over Time")
            
            # Daily message count
            daily_counts = analytics_df.set_index('timestamp').resample('D').size()
            fig_daily = px.line(daily_counts, title="Daily Message Volume", 
                               labels={'value': 'Messages', 'timestamp': 'Date'})
            st.plotly_chart(fig_daily, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Top Users by Activity")
            
            user_counts = analytics_df['user'].value_counts().head(10)
            fig_users = px.bar(user_counts, orientation='h', 
                             title="Most Active Users",
                             labels={'value': 'Message Count', 'index': 'User'})
            st.plotly_chart(fig_users, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Hourly Activity Heatmap and Common Words
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Hourly Activity Heatmap")
            
            # Create hourly heatmap data
            analytics_df['hour'] = analytics_df['timestamp'].dt.hour
            analytics_df['day'] = analytics_df['timestamp'].dt.date
            
            heatmap_data = analytics_df.groupby(['day', 'hour']).size().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(heatmap_data.T, 
                                  labels=dict(x="Date", y="Hour", color="Messages"),
                                  title="Message Activity by Hour and Date",
                                  aspect="auto")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Top Common Words")
            
            # Extract and display common words
            common_words = self.extract_common_topics(analytics_df['message'])
            if common_words:
                word_df = pd.DataFrame(common_words[:15], columns=['Word'])
                word_df['Rank'] = range(1, len(word_df) + 1)
                
                fig_words = px.bar(word_df.head(10), x='Word', y='Rank', 
                                 orientation='h', title="Most Frequent Words",
                                 labels={'Word': 'Word', 'Rank': 'Frequency Rank'})
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.info("No common words detected")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 3: Tickets Analysis and Conversation Length
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Tickets and Issues Analysis")
            
            tickets_count, ticket_samples = self.detect_tickets_raised(analytics_df['message'])
            
            st.metric("Potential Tickets Detected", tickets_count)
            
            if ticket_samples:
                st.write("**Sample ticket-related messages:**")
                for i, ticket_msg in enumerate(ticket_samples[:3], 1):
                    st.write(f"{i}. {ticket_msg[:100]}...")
            else:
                st.info("No ticket-related messages detected")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Conversation Patterns")
            
            # Message length distribution
            analytics_df['message_length'] = analytics_df['message'].str.len()
            fig_length = px.histogram(analytics_df, x='message_length', 
                                    title="Message Length Distribution",
                                    labels={'message_length': 'Message Length (chars)'})
            st.plotly_chart(fig_length, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">Enterprise Chat Analyzer</h1>', unsafe_allow_html=True)
    
    with col2:
        # Status indicator
        if SUMMARIZER_AVAILABLE:
            model_info = {
                "fast": ("Fast", "status-fast"),
                "balanced": ("Balanced", "status-balanced"),
                "quality": ("Quality", "status-quality")
            }
            name, cls = model_info.get(st.session_state.model_choice, ("AI", "status-balanced"))
            st.markdown(f'<div class="status-badge {cls}">{name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-mock">Mock Mode</div>', unsafe_allow_html=True)
    
    # Initialize app
    app = EnhancedChatApp()
    
    # Sidebar
    with st.sidebar:
        st.markdown('### Control Panel')
        
        # File upload
        uploaded_file = st.file_uploader("Upload Chat File", type=["csv", "json"])
        
        if uploaded_file:
            app.raw_chats = app.parse_uploaded_file(uploaded_file)
            if app.raw_chats:
                st.success(f"Loaded {len(app.raw_chats)} messages")
        
        # Sample data
        if not app.raw_chats:
            if st.button("Load Sample Data", use_container_width=True):
                app.raw_chats = [
                    {"timestamp": "2024-01-15T10:30:00", "user": "Alice", "message": "Let's discuss the Q1 project timeline. We need to finalize deliverables."},
                    {"timestamp": "2024-01-15T10:35:00", "user": "Bob", "message": "I agree. We should schedule a meeting with the development team this week."},
                    {"timestamp": "2024-01-15T10:40:00", "user": "Charlie", "message": "I'll prepare the initial documentation and share it by Wednesday."},
                    {"timestamp": "2024-01-15T10:45:00", "user": "Alice", "message": "Great! Let's aim for Thursday presentation to stakeholders."},
                    {"timestamp": "2024-01-15T10:50:00", "user": "Bob", "message": "I'll coordinate with the design team for the presentation materials."},
                ]
                st.success("Sample data loaded")
        
        st.markdown("---")
        view_mode = st.radio("**View Mode**", 
                           ["Chat Analysis", "Documentation", "Analytics"])
    
    # Main content
    if app.raw_chats:
        try:
            cleaned_chats = preprocess_chats(app.raw_chats)
            app.df = pd.DataFrame(cleaned_chats)
            app.df['timestamp'] = pd.to_datetime(app.df['timestamp'])
            
            # Date filter
            dates = ["All Dates"] + sorted(app.df['timestamp'].dt.date.unique())
            selected_date = st.selectbox("Filter by Date", dates)
            date_to_show = None if selected_date == "All Dates" else selected_date
            
            # Display selected view
            if view_mode == "Chat Analysis":
                app.display_chat_interface(app.raw_chats, date_to_show)
            elif view_mode == "Documentation":
                app.display_documentation_mode()
            else:
                app.display_analytics_dashboard(app.df, date_to_show)
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Upload a chat file or load sample data to begin analysis")

if __name__ == "__main__":
    main()