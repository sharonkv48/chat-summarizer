import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import json
import sys
import os
from fpdf import FPDF
import re
from typing import Dict, List, Tuple
import requests

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
    page_title="Enterprise Chat Analyzer Pro",
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
if 'integration_settings' not in st.session_state:
    st.session_state.integration_settings = {
        'jira_enabled': False,
        'slack_enabled': False,
        'salesforce_enabled': False
    }
if 'action_items' not in st.session_state:
    st.session_state.action_items = []
if 'jira_tickets_pending' not in st.session_state:
    st.session_state.jira_tickets_pending = []
if 'jira_tickets_approved' not in st.session_state:
    st.session_state.jira_tickets_approved = []
if 'date_range_type' not in st.session_state:
    st.session_state.date_range_type = "single_day"
if 'selected_start_date' not in st.session_state:
    st.session_state.selected_start_date = None
if 'selected_end_date' not in st.session_state:
    st.session_state.selected_end_date = None
if 'selected_single_date' not in st.session_state:
    st.session_state.selected_single_date = None
if 'filtered_messages' not in st.session_state:
    st.session_state.filtered_messages = []

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
    
    .action-item-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #f59e0b;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .jira-ticket-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .jira-pending { border-left: 4px solid #f59e0b; }
    .jira-approved { border-left: 4px solid #10b981; }
    
    .date-range-selector {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .feature-toggle-info {
        background: #f0f9ff;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 4px solid #0ea5e9;
        margin: 0.5rem 0;
        font-size: 0.9rem;
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

class AdvancedFeatures:
    """Class to handle all advanced features"""
    
    @staticmethod
    def extract_action_items(messages: List[str]) -> List[Dict]:
        """Extract action items, owners, and deadlines from messages"""
        action_items = []
        action_patterns = [
            r'(need to|must|should|will)\s+(.+)',
            r'(action item|todo|task):?\s*(.+)',
            r'(assign(?:ed)? to|owner:?)\s*(\w+).*?(due|by|deadline)?\s*(\d{4}-\d{2}-\d{2})?',
        ]
        
        for i, message in enumerate(messages):
            message_lower = message.lower()
            
            # Extract potential action items
            for pattern in action_patterns:
                matches = re.finditer(pattern, message_lower)
                for match in matches:
                    action_text = match.group()
                    action_items.append({
                        'id': len(action_items) + 1,
                        'action': action_text,
                        'owner': AdvancedFeatures.extract_owner(message),
                        'deadline': AdvancedFeatures.extract_deadline(message),
                        'priority': AdvancedFeatures.assess_priority(message),
                        'source_message': message[:100] + '...' if len(message) > 100 else message,
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
        
        return action_items
    
    @staticmethod
    def extract_owner(message: str) -> str:
        """Extract potential owner from message"""
        # Simple pattern matching for owners
        patterns = [
            r'(\w+)\s+will\s+',
            r'assign(?:ed)?\s+to\s+(\w+)',
            r'@(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Unassigned"
    
    @staticmethod
    def extract_deadline(message: str) -> str:
        """Extract deadlines from message"""
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(next week|tomorrow|end of day|EOD|EOB)',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        return "No deadline"
    
    @staticmethod
    def assess_priority(message: str) -> str:
        """Assess priority based on message content"""
        urgent_keywords = ['urgent', 'asap', 'immediately', 'critical', 'important']
        if any(keyword in message.lower() for keyword in urgent_keywords):
            return "High"
        return "Medium"
    
    @staticmethod
    def analyze_sentiment(messages: List[str]) -> Dict:
        """Analyze sentiment of messages"""
        try:
            sia = SentimentIntensityAnalyzer()
            sentiments = []
            
            for message in messages:
                score = sia.polarity_scores(message)
                sentiments.append(score)
            
            avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments)
            
            # Categorize overall sentiment
            if avg_compound >= 0.05:
                mood = "Positive"
            elif avg_compound <= -0.05:
                mood = "Negative"
            else:
                mood = "Neutral"
            
            return {
                'overall_sentiment': mood,
                'compound_score': avg_compound,
                'positive_percent': sum(1 for s in sentiments if s['compound'] > 0.05) / len(sentiments) * 100,
                'negative_percent': sum(1 for s in sentiments if s['compound'] < -0.05) / len(sentiments) * 100,
                'sentiment_trend': AdvancedFeatures.calculate_sentiment_trend(sentiments)
            }
        except:
            return {'overall_sentiment': 'Neutral', 'compound_score': 0, 'error': 'Sentiment analysis failed'}
    
    @staticmethod
    def calculate_sentiment_trend(sentiments: List[Dict]) -> str:
        """Calculate sentiment trend over time"""
        if len(sentiments) < 2:
            return "Stable"
        
        first_half = sum(s['compound'] for s in sentiments[:len(sentiments)//2]) / (len(sentiments)//2)
        second_half = sum(s['compound'] for s in sentiments[len(sentiments)//2:]) / (len(sentiments) - len(sentiments)//2)
        
        if second_half - first_half > 0.1:
            return "Improving"
        elif second_half - first_half < -0.1:
            return "Deteriorating"
        return "Stable"
    
    @staticmethod
    def detect_topics(messages: List[str]) -> List[Dict]:
        """Auto-categorize conversations into topics"""
        topic_keywords = {
            'Support Issues': ['error', 'problem', 'help', 'support', 'issue', 'bug', 'fix'],
            'Product Feedback': ['feedback', 'suggestion', 'improve', 'feature', 'idea'],
            'Feature Requests': ['request', 'need', 'want', 'would like', 'could we'],
            'Billing': ['payment', 'invoice', 'bill', 'charge', 'price', 'cost'],
            'Technical': ['technical', 'code', 'implementation', 'API', 'integration'],
            'Sales': ['sale', 'buy', 'purchase', 'demo', 'trial', 'price quote']
        }
        
        topic_counts = {topic: 0 for topic in topic_keywords.keys()}
        
        for message in messages:
            message_lower = message.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    topic_counts[topic] += 1
        
        return [{'topic': topic, 'count': count} 
                for topic, count in topic_counts.items() if count > 0]
    
    @staticmethod
    def detect_compliance_issues(messages: List[str]) -> List[Dict]:
        """Flag sensitive information and compliance violations"""
        pii_patterns = {
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'Credit Card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        }
        
        compliance_issues = []
        
        for i, message in enumerate(messages):
            for issue_type, pattern in pii_patterns.items():
                if re.search(pattern, message):
                    compliance_issues.append({
                        'type': issue_type,
                        'message_index': i,
                        'snippet': message[:100] + '...',
                        'severity': 'High'
                    })
        
        return compliance_issues
    
    @staticmethod
    def calculate_health_score(messages: List[str], sentiment: Dict) -> Dict:
        """Calculate conversation health score"""
        # Factors: sentiment, clarity, resolution, engagement
        score = 50  # Base score
        
        # Sentiment contribution
        sentiment_score = (sentiment.get('compound_score', 0) + 1) * 25  # 0-50 points
        score += sentiment_score
        
        # Message clarity (simple heuristic based on length and structure)
        avg_length = sum(len(msg) for msg in messages) / len(messages)
        clarity_score = min(avg_length / 50 * 10, 10)  # Up to 10 points
        score += clarity_score
        
        # Resolution indicators
        resolution_keywords = ['solved', 'resolved', 'completed', 'finished', 'done']
        resolution_count = sum(1 for msg in messages 
                             if any(keyword in msg.lower() for keyword in resolution_keywords))
        resolution_score = min(resolution_count / len(messages) * 20, 20)  # Up to 20 points
        score += resolution_score
        
        return {
            'overall_score': min(max(score, 0), 100),
            'sentiment_score': sentiment_score,
            'clarity_score': clarity_score,
            'resolution_score': resolution_score,
            'grade': 'A' if score >= 90 else 'B' if score >= 75 else 'C' if score >= 60 else 'D'
        }

class IntegrationManager:
    """Manage integrations with external tools"""
    
    @staticmethod
    def create_jira_ticket(summary: str, description: str, priority: str = "Medium") -> Dict:
        """Create Jira ticket (mock implementation)"""
        try:
            ticket_id = f"PROJ-{random.randint(1000, 9999)}"
            ticket = {
                'id': ticket_id,
                'summary': summary,
                'description': description,
                'priority': priority,
                'status': 'Pending Approval',
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'assignee': 'Unassigned'
            }
            return ticket
        except:
            return None
    
    @staticmethod
    def send_slack_message(channel: str, message: str) -> bool:
        """Send Slack message (mock implementation)"""
        try:
            # In real implementation, this would use Slack API
            st.success(f"Slack message sent to {channel}")
            return True
        except:
            st.error("Failed to send Slack message")
            return False

class EnhancedChatApp:
    def __init__(self):
        self.raw_chats = []
        self.df = pd.DataFrame()
        self.advanced_features = AdvancedFeatures()
        self.integration_manager = IntegrationManager()
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
    
    def get_filtered_messages(self, chats, date_range_type="single_day", selected_date=None, start_date=None, end_date=None):
        """Get filtered messages based on date range selection"""
        if date_range_type == "single_day":
            filtered_chats = [chat for chat in chats 
                             if not selected_date or 
                             pd.to_datetime(chat['timestamp']).date() == selected_date]
        else:  # date_range
            filtered_chats = [chat for chat in chats 
                             if start_date and end_date and
                             start_date <= pd.to_datetime(chat['timestamp']).date() <= end_date]
        
        return [chat['message'] for chat in filtered_chats], filtered_chats
    
    def get_chat_metrics(self, chats, date_range_type="single_day", selected_date=None, start_date=None, end_date=None):
        """Calculate chat metrics with date range support"""
        if date_range_type == "single_day":
            filtered_chats = [chat for chat in chats 
                             if not selected_date or 
                             pd.to_datetime(chat['timestamp']).date() == selected_date]
        else:  # date_range
            filtered_chats = [chat for chat in chats 
                             if start_date and end_date and
                             start_date <= pd.to_datetime(chat['timestamp']).date() <= end_date]
        
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
        
        date_range_text = ""
        if timestamps:
            if date_range_type == "single_day":
                date_range_text = f"Single day: {min(timestamps).strftime('%Y-%m-%d')}"
            else:
                date_range_text = f"{min(timestamps).strftime('%Y-%m-%d')} to {max(timestamps).strftime('%Y-%m-%d')}"
        
        return {
            'total_messages': total_messages,
            'unique_users': unique_users,
            'avg_message_length': round(avg_message_length, 1),
            'time_span_hours': round(time_span.total_seconds() / 3600, 1),
            'common_topics': common_words[:10],
            'date_range': date_range_text,
            'filtered_chats': filtered_chats
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
    
    def display_date_range_selector(self):
        """Display date range selection interface"""
        st.markdown('<div class="date-range-selector">', unsafe_allow_html=True)
        st.markdown("### Date Range Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_range_type = st.radio(
                "Select Date Range Type:",
                ["single_day", "date_range"],
                format_func=lambda x: "Single Day" if x == "single_day" else "Date Range",
                key="date_range_type_selector"
            )
            
            if date_range_type == "single_day":
                # Single date selector
                if self.df is not None and not self.df.empty:
                    min_date = self.df['timestamp'].min().date()
                    max_date = self.df['timestamp'].max().date()
                    selected_date = st.date_input(
                        "Select Date:",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="single_date_selector"
                    )
                    st.session_state.selected_single_date = selected_date
                    st.session_state.selected_start_date = None
                    st.session_state.selected_end_date = None
                else:
                    st.info("No data available for date selection")
                    selected_date = None
            
            else:  # date_range
                # Date range selector
                if self.df is not None and not self.df.empty:
                    min_date = self.df['timestamp'].min().date()
                    max_date = self.df['timestamp'].max().date()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            "Start Date:",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="start_date_selector"
                        )
                    with col2:
                        end_date = st.date_input(
                            "End Date:",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="end_date_selector"
                        )
                    
                    if start_date > end_date:
                        st.error("End date must be after start date")
                        start_date, end_date = end_date, start_date
                    
                    st.session_state.selected_start_date = start_date
                    st.session_state.selected_end_date = end_date
                    st.session_state.selected_single_date = None
                    
                    # Display selected range info
                    st.info(f"Selected range: {start_date} to {end_date}")
                else:
                    st.info("No data available for date range selection")
        
        with col2:
            # Display date statistics
            if self.df is not None and not self.df.empty:
                st.markdown("**Date Statistics:**")
                total_days = (self.df['timestamp'].max() - self.df['timestamp'].min()).days + 1
                st.write(f"• Total days in data: {total_days}")
                st.write(f"• Date range: {self.df['timestamp'].min().date()} to {self.df['timestamp'].max().date()}")
                st.write(f"• Total messages: {len(self.df)}")
                
                # Show message distribution by date
                if total_days > 1:
                    daily_counts = self.df.set_index('timestamp').resample('D').size()
                    st.write(f"• Average messages per day: {daily_counts.mean():.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return date_range_type
    
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
    
    def display_chat_interface(self, chats):
        """Main chat analysis interface"""
        # Date range selection
        date_range_type = self.display_date_range_selector()
        
        # Get filtered chats based on date selection
        if date_range_type == "single_day":
            selected_date = st.session_state.selected_single_date
            start_date = end_date = None
        else:
            selected_date = None
            start_date = st.session_state.selected_start_date
            end_date = st.session_state.selected_end_date
        
        # Get filtered messages for feature toggles
        filtered_messages, filtered_chats = self.get_filtered_messages(
            chats, 
            date_range_type=date_range_type,
            selected_date=selected_date,
            start_date=start_date,
            end_date=end_date
        )
        
        # Store filtered messages in session state for feature toggles
        st.session_state.filtered_messages = filtered_messages
        
        # Get metrics with the correct parameters
        metrics = self.get_chat_metrics(
            chats, 
            date_range_type=date_range_type,
            selected_date=selected_date,
            start_date=start_date,
            end_date=end_date
        )
        
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
        date_range_text = metrics.get('date_range', 'All Dates')
        self.display_chat_messages(filtered_chats, date_range_text)
        
        # Metrics
        if filtered_chats:
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
    
    def display_chat_messages(self, chats, date_range_text):
        """Display chat messages"""
        st.markdown(f'''
        <div class="chat-container">
            <div class="chat-header">
                <div>
                    <h3 style="margin:0;">Live Chat Window</h3>
                    <small>{date_range_text} • {len(chats)} messages</small>
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
    
    def display_analytics_dashboard(self, df):
        """Enhanced Analytics dashboard with insights"""
        st.markdown("### Analytics Dashboard")
        
        # Date range selection for analytics
        date_range_type = self.display_date_range_selector()
        
        if date_range_type == "single_day":
            selected_date = st.session_state.selected_single_date
            if selected_date and not df.empty:
                analytics_df = df[df['timestamp'].dt.date == selected_date]
            else:
                analytics_df = df
        else:
            start_date = st.session_state.selected_start_date
            end_date = st.session_state.selected_end_date
            if start_date and end_date and not df.empty:
                analytics_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
            else:
                analytics_df = df
        
        if analytics_df.empty:
            st.info("No data available for selected date range")
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
            if len(analytics_df) > 0:
                date_range_text = f"{analytics_df['timestamp'].min().date()} to {analytics_df['timestamp'].max().date()}"
            else:
                date_range_text = "No data"
            st.metric("Date Range", date_range_text)
        
        # Row 1: Message Frequency and Top Users
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Message Frequency Over Time")
            
            # Daily message count
            if len(analytics_df) > 0:
                daily_counts = analytics_df.set_index('timestamp').resample('D').size()
                fig_daily = px.line(daily_counts, title="Daily Message Volume", 
                                   labels={'value': 'Messages', 'timestamp': 'Date'})
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("No data available for chart")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Top Users by Activity")
            
            if len(analytics_df) > 0:
                user_counts = analytics_df['user'].value_counts().head(10)
                fig_users = px.bar(user_counts, orientation='h', 
                                 title="Most Active Users",
                                 labels={'value': 'Message Count', 'index': 'User'})
                st.plotly_chart(fig_users, use_container_width=True)
            else:
                st.info("No data available for chart")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Row 2: Hourly Activity Heatmap and Common Words
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Hourly Activity Heatmap")
            
            if len(analytics_df) > 0:
                # Create hourly heatmap data
                analytics_df['hour'] = analytics_df['timestamp'].dt.hour
                analytics_df['day'] = analytics_df['timestamp'].dt.date
                
                heatmap_data = analytics_df.groupby(['day', 'hour']).size().unstack(fill_value=0)
                
                fig_heatmap = px.imshow(heatmap_data.T, 
                                      labels=dict(x="Date", y="Hour", color="Messages"),
                                      title="Message Activity by Hour and Date",
                                      aspect="auto")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No data available for chart")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-section">', unsafe_allow_html=True)
            st.markdown("#### Top Common Words")
            
            if len(analytics_df) > 0:
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
            else:
                st.info("No data available for analysis")
            st.markdown('</div>', unsafe_allow_html=True)

    # UPDATED FEATURE TOGGLE METHODS - NOW RESPECT DATE FILTERS
    def display_action_items_panel(self, messages=None):
        """Display action items extraction panel with date filter support"""
        st.markdown("### Action Items & Task Extraction")
        
        # Use filtered messages if provided, otherwise use session state filtered messages
        if messages is None:
            messages = st.session_state.filtered_messages
        
        if not messages:
            st.info("No messages available for the selected date range")
            return
        
        # Show filter info
        st.markdown(f'<div class="feature-toggle-info">📅 Analyzing {len(messages)} messages from selected date range</div>', unsafe_allow_html=True)
        
        # Extract action items from filtered messages
        action_items = self.advanced_features.extract_action_items(messages)
        
        if action_items:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for i, item in enumerate(action_items):
                    with st.expander(f"Action {i+1}: {item['action'][:50]}...", expanded=True):
                        cola, colb, colc = st.columns([2, 2, 2])
                        with cola: 
                            st.write(f"**Owner:** {item['owner']}")
                        with colb: 
                            st.write(f"**Deadline:** {item['deadline']}")
                        with colc: 
                            st.write(f"**Priority:** {item['priority']}")
                        
                        st.write(f"**Source:** {item['source_message']}")
                        st.write(f"**Created:** {item['created_at']}")
            
            with col2:
                st.markdown("**Integrations**")
                
                # Checkbox to select action items for Jira creation
                st.markdown("**Select for Jira:**")
                selected_items = []
                for i, item in enumerate(action_items):
                    if st.checkbox(f"Action {i+1}", key=f"select_{i}"):
                        selected_items.append(item)
                
                if selected_items and st.button("Create Jira Tickets", key="jira_create"):
                    for item in selected_items:
                        ticket = self.integration_manager.create_jira_ticket(
                            item['action'], 
                            item['source_message'],
                            item['priority']
                        )
                        if ticket:
                            st.session_state.jira_tickets_pending.append(ticket)
                            st.success(f"Created Jira ticket: {ticket['id']}")
                
                if st.button("Send Slack Reminders", key="slack_remind"):
                    for item in action_items:
                        self.integration_manager.send_slack_message(
                            "general", 
                            f"Reminder: {item['action']} - Owner: {item['owner']}"
                        )
        else:
            st.info("No action items detected in the selected date range")
    
    def display_sentiment_analysis(self, messages=None):
        """Display sentiment analysis results with date filter support"""
        st.markdown("### Sentiment & Emotion Analysis")
        
        # Use filtered messages if provided, otherwise use session state filtered messages
        if messages is None:
            messages = st.session_state.filtered_messages
        
        if not messages:
            st.info("No messages available for the selected date range")
            return
        
        # Show filter info
        st.markdown(f'<div class="feature-toggle-info">📅 Analyzing {len(messages)} messages from selected date range</div>', unsafe_allow_html=True)
        
        sentiment_results = self.advanced_features.analyze_sentiment(messages)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Sentiment", 
                sentiment_results['overall_sentiment'],
                delta=f"{sentiment_results['compound_score']:.2f} compound"
            )
        
        with col2:
            st.metric("Positive Messages", f"{sentiment_results.get('positive_percent', 0):.1f}%")
        
        with col3:
            st.metric("Negative Messages", f"{sentiment_results.get('negative_percent', 0):.1f}%")
        
        with col4:
            st.metric("Trend", sentiment_results.get('sentiment_trend', 'Stable'))
        
        # Sentiment over time chart
        if len(messages) > 1:
            try:
                sia = SentimentIntensityAnalyzer()
                sentiment_scores = [sia.polarity_scores(msg)['compound'] for msg in messages]
                
                fig = px.line(
                    x=range(len(sentiment_scores)),
                    y=sentiment_scores,
                    title="Sentiment Trend Over Selected Date Range",
                    labels={'x': 'Message Sequence', 'y': 'Sentiment Score'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Sentiment trend analysis requires more messages")
    
    def display_topic_analysis(self, messages=None):
        """Display topic detection results with date filter support"""
        st.markdown("### Topic & Trend Detection")
        
        # Use filtered messages if provided, otherwise use session state filtered messages
        if messages is None:
            messages = st.session_state.filtered_messages
        
        if not messages:
            st.info("No messages available for the selected date range")
            return
        
        # Show filter info
        st.markdown(f'<div class="feature-toggle-info">📅 Analyzing {len(messages)} messages from selected date range</div>', unsafe_allow_html=True)
        
        topics = self.advanced_features.detect_topics(messages)
        
        if topics:
            # Topic distribution chart
            topic_df = pd.DataFrame(topics)
            fig = px.pie(
                topic_df, 
                values='count', 
                names='topic',
                title="Conversation Topics Distribution (Selected Date Range)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic trends over time (simplified)
            st.write("**Topic Frequency:**")
            for topic in topics:
                st.write(f"- {topic['topic']}: {topic['count']} mentions")
        else:
            st.info("No specific topics detected in the selected date range")
    
    def display_compliance_analysis(self, messages=None):
        """Display compliance and risk detection results with date filter support"""
        st.markdown("### Compliance & Risk Detection")
        
        # Use filtered messages if provided, otherwise use session state filtered messages
        if messages is None:
            messages = st.session_state.filtered_messages
        
        if not messages:
            st.info("No messages available for the selected date range")
            return
        
        # Show filter info
        st.markdown(f'<div class="feature-toggle-info">📅 Analyzing {len(messages)} messages from selected date range</div>', unsafe_allow_html=True)
        
        compliance_issues = self.advanced_features.detect_compliance_issues(messages)
        
        if compliance_issues:
            st.warning(f"⚠️ {len(compliance_issues)} potential compliance issues detected in selected date range")
            
            for issue in compliance_issues:
                st.error(
                    f"**{issue['type']}** (Severity: {issue['severity']})\n"
                    f"Message {issue['message_index'] + 1}: {issue['snippet']}"
                )
        else:
            st.success("✅ No compliance issues detected in selected date range")
    
    def display_health_score(self, messages=None, sentiment_results=None):
        """Display conversation health score with date filter support"""
        st.markdown("### Conversation Health Score")
        
        # Use filtered messages if provided, otherwise use session state filtered messages
        if messages is None:
            messages = st.session_state.filtered_messages
        
        if not messages:
            st.info("No messages available for the selected date range")
            return
        
        # Show filter info
        st.markdown(f'<div class="feature-toggle-info">📅 Analyzing {len(messages)} messages from selected date range</div>', unsafe_allow_html=True)
        
        # Calculate sentiment if not provided
        if sentiment_results is None:
            sentiment_results = self.advanced_features.analyze_sentiment(messages)
        
        health_score = self.advanced_features.calculate_health_score(messages, sentiment_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Health score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score['overall_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Sentiment Score", f"{health_score['sentiment_score']:.1f}")
        
        with col3:
            st.metric("Clarity Score", f"{health_score['clarity_score']:.1f}")
        
        with col4:
            st.metric("Resolution Score", f"{health_score['resolution_score']:.1f}")
        
        st.write(f"**Overall Grade:** {health_score['grade']}")

    def display_jira_tickets_section(self):
        """Display Jira tickets management section"""
        st.markdown("### Jira Tickets Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pending Approval")
            if not st.session_state.jira_tickets_pending:
                st.info("No tickets pending approval")
            else:
                for i, ticket in enumerate(st.session_state.jira_tickets_pending):
                    with st.container():
                        st.markdown(f'<div class="jira-ticket-card jira-pending">', unsafe_allow_html=True)
                        st.write(f"**{ticket['id']}** - {ticket['summary'][:50]}...")
                        st.write(f"Priority: {ticket['priority']} | Status: {ticket['status']}")
                        st.write(f"Created: {ticket['created_at']}")
                        
                        cola, colb = st.columns(2)
                        with cola:
                            if st.button("Approve", key=f"approve_{ticket['id']}"):
                                ticket['status'] = 'Approved'
                                ticket['approved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                                st.session_state.jira_tickets_approved.append(ticket)
                                st.session_state.jira_tickets_pending.remove(ticket)
                                st.rerun()
                        with colb:
                            if st.button("Reject", key=f"reject_{ticket['id']}"):
                                st.session_state.jira_tickets_pending.remove(ticket)
                                st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Approved Tickets")
            if not st.session_state.jira_tickets_approved:
                st.info("No approved tickets")
            else:
                for ticket in st.session_state.jira_tickets_approved:
                    with st.container():
                        st.markdown(f'<div class="jira-ticket-card jira-approved">', unsafe_allow_html=True)
                        st.write(f"**{ticket['id']}** - {ticket['summary'][:50]}...")
                        st.write(f"Priority: {ticket['priority']} | Status: {ticket['status']}")
                        st.write(f"Approved: {ticket.get('approved_at', 'N/A')}")
                        st.markdown('</div>', unsafe_allow_html=True)

def fetch_slack_messages(token, channel_id, limit=1000):
    """
    Fetch messages from Slack channel using API
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    url = f"https://slack.com/api/conversations.history"
    params = {
        "channel": channel_id,
        "limit": limit
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("ok"):
            error_msg = data.get("error", "Unknown error")
            st.error(f"Slack API Error: {error_msg}")
            return None
            
        messages = data.get("messages", [])
        
        # Convert Slack format to your app's format
        converted_messages = []
        for msg in messages:
            # Skip bot messages and system messages if desired
            if msg.get("subtype") in ["bot_message", "channel_join", "channel_leave"]:
                continue
                
            converted_msg = {
                "timestamp": datetime.fromtimestamp(float(msg["ts"])).isoformat(),
                "user": msg.get("user", "Unknown"),
                "message": msg.get("text", "")
            }
            converted_messages.append(converted_msg)
            
        return converted_messages
        
    except requests.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching Slack messages: {str(e)}")
        return None

def main():
    # Header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">Enterprise Chat Analyzer Pro</h1>', unsafe_allow_html=True)
    
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
        
        # Data source selection
        data_source = st.radio(
            "**Data Source**", 
            ["Upload File", "Slack API", "Sample Data"],
            help="Choose how to load your chat data"
        )
        
        st.markdown("---")
        
        # File upload option
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload Chat File", type=["csv", "json"])
            
            if uploaded_file:
                app.raw_chats = app.parse_uploaded_file(uploaded_file)
                if app.raw_chats:
                    st.success(f"Loaded {len(app.raw_chats)} messages")
        
        # Slack API option
        elif data_source == "Slack API":
            st.markdown("#### Slack Configuration")
            
            # Input fields for Slack credentials
            slack_token = st.text_input(
                "Slack Bot Token", 
                type="password",
                placeholder="xoxb-...",
                help="Your Slack bot token (starts with xoxb-)"
            )
            
            channel_id = st.text_input(
                "Channel ID", 
                placeholder="C09GN1EHH6W",
                help="Slack channel ID to fetch messages from"
            )
            
            message_limit = st.number_input(
                "Message Limit", 
                min_value=10, 
                max_value=1000, 
                value=100,
                help="Number of recent messages to fetch"
            )
            
            # Fetch button
            if st.button("Fetch from Slack", use_container_width=True):
                if not slack_token or not channel_id:
                    st.error("Please provide both Slack token and channel ID")
                else:
                    with st.spinner("Fetching messages from Slack..."):
                        fetched = fetch_slack_messages(slack_token, channel_id, message_limit)
                        app.raw_chats = fetched if fetched else []

                    if app.raw_chats:
                        st.success(f"Loaded {len(app.raw_chats)} messages from Slack")
                        # Persist into session state so it survives reruns
                        st.session_state.slack_data = app.raw_chats
                        # Ensure the main area picks up the new data immediately
                        try:
                            st.experimental_rerun()
                        except Exception:
                            # If Streamlit isn't running in the usual session (e.g. running `python app.py`),
                            # experimental_rerun can raise. Ignore in that case but inform user.
                            st.info("Data loaded — please refresh the app to see messages if automatic rerun isn't available")
            
            # Load from session state if available (persistence across reruns)
            if hasattr(st.session_state, 'slack_data') and (not app.raw_chats or len(app.raw_chats) == 0):
                app.raw_chats = st.session_state.slack_data or []
                st.info(f"Using previously loaded Slack data ({len(app.raw_chats)} messages)")
        
        # Sample data option
        elif data_source == "Sample Data":
            if st.button("Load Sample Data", use_container_width=True):
                app.raw_chats = [
                    {"timestamp": "2024-01-15T10:30:00", "user": "Alice", "message": "We need to fix the login issue urgently. Assign to Bob due by 2024-01-20."},
                    {"timestamp": "2024-01-15T10:35:00", "user": "Bob", "message": "I'll work on it. Need to check the API integration first."},
                    {"timestamp": "2024-01-15T10:40:00", "user": "Charlie", "message": "Customer reported billing problem with invoice INV-12345."},
                    {"timestamp": "2024-01-15T10:45:00", "user": "Alice", "message": "This is critical for our Q1 goals. Please prioritize."},
                    {"timestamp": "2024-01-15T10:50:00", "user": "Bob", "message": "Understood. I've created a ticket and will update by EOD."},
                    {"timestamp": "2024-01-16T09:15:00", "user": "Alice", "message": "Follow up on the login issue progress."},
                    {"timestamp": "2024-01-16T09:20:00", "user": "Bob", "message": "Fixed the API authentication bug. Testing now."},
                    {"timestamp": "2024-01-17T14:30:00", "user": "Charlie", "message": "New feature request from customer: dark mode toggle."},
                    {"timestamp": "2024-01-17T14:45:00", "user": "Alice", "message": "Add to backlog for next sprint planning."},
                    {"timestamp": "2024-01-18T11:00:00", "user": "Bob", "message": "Login issue resolved and deployed to production."},
                ]
                st.success("Sample data loaded")
        
        st.markdown("---")
        
        # Additional options when data is loaded
        if app.raw_chats:
            # Data info
            st.markdown(f"**Messages loaded:** {len(app.raw_chats)}")
            
            # Export option
            if st.button("Export Current Data", use_container_width=True):
                export_data = json.dumps(app.raw_chats, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"chat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.markdown("---")
        st.markdown("### Feature Toggles")
        st.markdown("*Features will analyze only the selected date range*")
        
        # Feature toggles
        show_action_items = st.checkbox("Action Item Extraction", value=True)
        show_sentiment = st.checkbox("Sentiment Analysis", value=True)
        show_topics = st.checkbox("Topic Detection", value=True)
        show_compliance = st.checkbox("Compliance Check", value=True)
        show_health = st.checkbox("Health Score", value=True)
        
        st.markdown("---")
        view_mode = st.radio("**View Mode**", 
                           ["Chat Analysis", "Documentation", "Jira Tickets", "Analytics"])
    
    # Main content
    if app.raw_chats:
        try:
            # Use the imported preprocess_chats function
            cleaned_chats = preprocess_chats(app.raw_chats)
            app.df = pd.DataFrame(cleaned_chats)
            app.df['timestamp'] = pd.to_datetime(app.df['timestamp'])
            
            if view_mode == "Chat Analysis":
                app.display_chat_interface(cleaned_chats)
                
                # Advanced features panels - now respect date filters
                if show_action_items:
                    app.display_action_items_panel()
                
                if show_sentiment:
                    app.display_sentiment_analysis()
                
                if show_topics:
                    app.display_topic_analysis()
                
                if show_compliance:
                    app.display_compliance_analysis()
                
                if show_health:
                    # Calculate sentiment once and reuse for health score
                    filtered_messages = st.session_state.filtered_messages
                    if filtered_messages:
                        sentiment_results = app.advanced_features.analyze_sentiment(filtered_messages)
                        app.display_health_score(filtered_messages, sentiment_results)
                    else:
                        app.display_health_score()
            
            elif view_mode == "Documentation":
                app.display_documentation_mode()
            
            elif view_mode == "Jira Tickets":
                app.display_jira_tickets_section()
            
            else:
                app.display_analytics_dashboard(app.df)
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        # Instructions based on selected data source
        if data_source == "Upload File":
            st.info("📁 Upload a chat file to begin analysis")
        elif data_source == "Slack API":
            st.info("🔗 Configure Slack credentials and fetch messages to begin analysis")
        else:
            st.info("📊 Load sample data to begin analysis")

if __name__ == "__main__":
    main()