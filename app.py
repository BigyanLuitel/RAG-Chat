import streamlit as st
from dotenv import load_dotenv
from core.answer import answer_question

load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="Orchid International College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e0e0e0;
        margin-top: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #667eea;
    }
    .context-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin-top: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
    }
    .sidebar-info {
        background-color: #f0f4c3;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #33691e;
    }
    /* Center the chat interface */
    .main .block-container {
        max-width: 800px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def format_context(context):
    """Format retrieved context for display"""
    if not context:
        return "No context retrieved for this query."
    
    result = ""
    for i, doc in enumerate(context, 1):
        source = doc.metadata.get('source', 'Unknown')
        result += f"**📄 Source {i}:** `{source}`\n\n"
        result += f"{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n\n"
        result += "---\n\n"
    return result


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = []
    if "awaiting_name" not in st.session_state:
        st.session_state.awaiting_name = False

def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(message["content"])


def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎓 Orchid International College")
        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-info">
        <h4>📚 About This Assistant</h4>
        <p>I'm your AI-powered assistant for Orchid International College. 
        Ask me anything about:</p>
        <ul>
            <li>📖 Academic Programs</li>
            <li>🎯 Admission Process</li>
            <li>🏫 Campus Facilities</li>
            <li>📞 Contact Information</li>
            <li>💼 Career Opportunities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Programs quick links
        st.markdown("### 🎯 Our Programs")
        programs = [
            "BSc CSIT", "BCA", "BITM", "BBM", "BBS", "BSW"
        ]
        for program in programs:
            st.markdown(f"• {program}")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.context = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📞 Quick Contact")
        st.markdown("📧 info@orchidcollege.edu.np")
        st.markdown("📱 +977-XXX-XXXXXXX")
    
    # Main content area
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Orchid International College</h1>
        <p>Your AI-Powered College Assistant - Ask me anything!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat section (centered via CSS)
    st.markdown("### 💬 Chat with Us")
    
    # Display chat history
    chat_container = st.container(height=450)
    with chat_container:
        display_chat_history()
        
        # Show welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
            👋 **Welcome to Orchid International College!**
            
            I'm here to help you with any questions about our college. You can ask me about:
            - Admission requirements and process
            - Academic programs (BSc CSIT, BCA, BITM, BBM, BBS, BSW)
            - Fee structure and scholarships
            - Campus facilities and location
            - Contact information
            
            Just type your question below to get started!
            """)
    
   
    
    # Chat input at the bottom
    if prompt := st.chat_input("Ask anything about Orchid International College..."):
        # Check if we're waiting for the user's name for admission booking
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from the model
        with st.spinner("🔍 Searching our knowledge base..."):
            try:
                # Get prior messages for context
                prior = st.session_state.messages[:-1]
                answer, context = answer_question(prompt, prior)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.context = context
                
            except Exception as e:
                import traceback
                st.error("Full error traceback:")
                st.code(traceback.format_exc())
                raise e
        
        st.rerun()


if __name__ == "__main__":
    main()
