"""Streamlit UI for Chat with PDF RAG application."""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from ingest import process_pdf, get_embeddings, load_chromadb
from rag_chain import query_pdf

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [role="tab"] {
        font-size: 1.1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "use_ollama" not in st.session_state:
    st.session_state.use_ollama = False
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-4o-mini"

# Header
st.title("📄 Chat with PDF - RAG Assistant")
st.markdown("Upload PDFs and ask questions with source citations")
st.divider()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("LLM Settings")
    use_ollama = st.checkbox(
        "Use Ollama (Local LLM)",
        value=st.session_state.use_ollama,
        help="Use local Ollama instead of OpenAI API"
    )
    st.session_state.use_ollama = use_ollama
    
    if use_ollama:
        ollama_model = st.text_input(
            "Ollama Model",
            value=os.getenv('OLLAMA_MODEL', 'llama2'),
            help="e.g., llama2, neural-chat"
        )
        ollama_base = st.text_input(
            "Ollama Base URL",
            value=os.getenv('OLLAMA_API_BASE', 'http://localhost:11434'),
            help="URL where Ollama is running"
        )
    else:
        llm_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state.llm_model = llm_model
        if not os.getenv('OPENAI_API_KEY'):
            st.warning("⚠️ OPENAI_API_KEY not set in environment")
    
    st.subheader("Retrieval Settings")
    k_chunks = st.slider(
        "Number of chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=4,
        help="Top-k similar chunks for context"
    )
    
    st.subheader("Embeddings")
    use_local_embeddings = st.checkbox(
        "Use Local Embeddings",
        value=os.getenv('USE_LOCAL_EMBEDDINGS', 'false').lower() == 'true',
        help="Use HuggingFace embeddings (free) vs OpenAI (costs)"
    )
    
    st.divider()
    st.subheader("📊 Status")
    if st.session_state.vectorstore:
        st.success("✅ PDF loaded and ready")
        st.metric("Chat Messages", len(st.session_state.chat_history))
    else:
        st.info("⏳ Upload a PDF to get started")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload academic papers or documents"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save temporarily
                        temp_path = f"/tmp/{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process PDF
                        vectorstore, chunks = process_pdf(temp_path)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        
                        st.success(f"✅ Processed {uploaded_file.name}")
                        st.info(f"Created {len(chunks)} chunks for retrieval")
                        
                        # Cleanup
                        os.remove(temp_path)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

with col2:
    st.subheader("📚 Quick Help")
    st.markdown("""
    **Tips:**
    - Ask specific questions
    - Questions about content work best
    - View retrieved chunks to verify sources
    - Out-of-scope questions will be flagged
    
    **Example Questions:**
    - What is the main topic?
    - What are key findings?
    - How does method A compare to B?
    """)

st.divider()

# Chat interface
if st.session_state.vectorstore:
    st.subheader("💬 Chat")
    
    # Display chat history
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            else:
                st.chat_message("assistant").markdown(message["content"])
    
    # Chat input
    user_question = st.chat_input(
        "Ask a question about the PDF...",
        key="chat_input"
    )
    
    if user_question:
        # Display user message
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        
        # Get answer
        with st.spinner("🔍 Retrieving context and generating answer..."):
            try:
                answer, sources = query_pdf(
                    question=user_question,
                    vectorstore=st.session_state.vectorstore,
                    use_ollama=st.session_state.use_ollama,
                    llm_model=st.session_state.llm_model if not st.session_state.use_ollama else "llama2",
                    k=k_chunks
                )
                
                # Display answer
                st.chat_message("assistant").markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                
                # Display source chunks in expandable section
                with st.expander(f"📚 View Retrieved Chunks ({len(sources)} found)", expanded=False):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"### Chunk {i} - Page {source['page']}, Paragraph {source['paragraph']}")
                        st.markdown(
                            f"<div class='source-box'>{source['text']}</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"Source: {source['source']} | ID: {source['chunk_id']}")
                        st.divider()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🔄 Upload New PDF"):
                st.session_state.vectorstore = None
                st.session_state.chat_history = []
                st.session_state.uploaded_files = []
                st.rerun()

else:
    st.info("👆 Upload a PDF to start asking questions!")
    
    st.markdown("""
    ## 🚀 Getting Started
    
    1. **Upload a PDF**: Use the file uploader on the left
    2. **Wait for processing**: The app will extract text and create embeddings
    3. **Ask questions**: Type your question and get answers with citations
    4. **Verify sources**: Expand the "View Retrieved Chunks" section to see context
    
    ## 🎯 Features
    
    ✅ **Source Citations**: Every answer cites Page X, Paragraph Y  
    ✅ **Multi-Chunk Reasoning**: Combines information from multiple parts  
    ✅ **Out-of-Scope Detection**: Explicitly says when info isn't in the document  
    ✅ **Local Processing**: ChromaDB stores embeddings locally  
    ✅ **Flexible LLM**: Use OpenAI GPT or local Ollama  
    
    ## 💡 Example Questions
    
    - "What is the main contribution of this paper?"
    - "How do the authors address previous limitations?"
    - "What are the experimental results on benchmark X?"
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; font-size: 0.9rem; color: gray;">
    <p>Chat with PDF - RAG Assistant | Built with LangChain, ChromaDB, and Streamlit</p>
    <p><a href="https://github.com/krisha-padariya/chat-with-pdf-rag">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
