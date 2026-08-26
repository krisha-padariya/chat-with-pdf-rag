"""RAG chain with citation-aware prompts."""

import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

# Safely load .env, ignore if corrupted
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️ Warning: Could not load .env file: {e}")
    print("Continuing without .env file...")


def get_citation_aware_prompt() -> PromptTemplate:
    """
    Create a prompt template that enforces source citations.
    
    Returns:
        PromptTemplate configured for citation-aware responses
    """
    template = """You are a helpful assistant that answers questions using ONLY the provided context from a document. 
Each context chunk includes its page number and paragraph number.

RULES:
1. Answer using only the given context — do not use outside knowledge.
2. After every claim, cite the exact page and paragraph it came from, like this: (Source: Page X, Paragraph Y)
3. If the context doesn't contain the answer, say: "I couldn't find this information in the document" — do not guess or fabricate.
4. If multiple context chunks support the answer, cite all of them.
5. Be concise but complete in your answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def get_llm(use_ollama: bool = False, model: str = "gpt-4o-mini"):
    """
    Initialize LLM based on configuration.
    
    Args:
        use_ollama: If True, use Ollama; if False, use OpenAI
        model: Model name
    
    Returns:
        LLM object
    
    Raises:
        ValueError: If API key missing or Ollama not running
    """
    if use_ollama:
        print("Using Ollama LLM...")
        base_url = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        return Ollama(
            model=ollama_model,
            base_url=base_url,
            temperature=0
        )
    else:
        print(f"Using OpenAI LLM: {model}")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0
        )


def create_rag_chain(
    vectorstore: Chroma,
    use_ollama: bool = False,
    llm_model: str = "gpt-4o-mini",
    k: int = 4
) -> Tuple:
    """
    Create RAG chain with retrieval and LLM.
    
    Args:
        vectorstore: ChromaDB vectorstore with embeddings
        use_ollama: Use Ollama instead of OpenAI
        llm_model: LLM model name
        k: Number of chunks to retrieve
    
    Returns:
        Tuple of (rag_chain, retriever)
    """
    llm = get_llm(use_ollama=use_ollama, model=llm_model)
    prompt = get_citation_aware_prompt()
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return rag_chain, retriever


def format_source_citations(result: Dict) -> Tuple[str, List[Dict]]:
    """
    Extract and format answer with citations and source documents.
    
    Args:
        result: Result dict from RAG chain with 'result' and 'source_documents'
    
    Returns:
        Tuple of (formatted_answer, source_chunks)
    """
    answer = result.get('result', 'No answer generated')
    source_docs = result.get('source_documents', [])
    
    # Extract metadata from source documents
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        sources.append({
            'text': doc.page_content,
            'page': int(metadata.get('page', 1)),
            'paragraph': int(metadata.get('paragraph', 1)),
            'source': metadata.get('source', 'unknown'),
            'chunk_id': metadata.get('chunk_id', 'unknown')
        })
    
    return answer, sources


def query_pdf(
    question: str,
    vectorstore: Chroma,
    use_ollama: bool = False,
    llm_model: str = "gpt-4o-mini",
    k: int = 4
) -> Tuple[str, List[Dict]]:
    """
    Query PDF and get answer with citations.
    
    Args:
        question: User question
        vectorstore: ChromaDB vectorstore
        use_ollama: Use Ollama instead of OpenAI
        llm_model: LLM model name
        k: Number of chunks to retrieve
    
    Returns:
        Tuple of (answer_with_citations, source_chunks)
    
    Raises:
        ValueError: If RAG chain fails
    """
    try:
        rag_chain, _ = create_rag_chain(
            vectorstore=vectorstore,
            use_ollama=use_ollama,
            llm_model=llm_model,
            k=k
        )
        
        result = rag_chain({"query": question})
        answer, sources = format_source_citations(result)
        
        return answer, sources
    
    except Exception as e:
        raise ValueError(f"Error querying PDF: {str(e)}")


if __name__ == "__main__":
    # Test prompt template
    prompt = get_citation_aware_prompt()
    print("\n📋 Citation-Aware Prompt Template:")
    print(prompt.template)
