"""PDF loading, chunking, embedding, and ChromaDB storage."""

import os
import json
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# Safely load .env, ignore if corrupted
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️ Warning: Could not load .env file: {e}")
    print("Continuing without .env file...")


def load_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Load PDF and extract text with page and paragraph metadata.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (full_text, list of paragraph dicts with metadata)
    
    Raises:
        ValueError: If PDF cannot be read
    """
    try:
        paragraphs = []
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if not page_text:
                    continue
                
                # Split page text into paragraphs (separated by double newlines)
                page_paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                
                for para_idx, paragraph in enumerate(page_paragraphs, 1):
                    paragraphs.append({
                        'text': paragraph,
                        'page': page_num,
                        'paragraph': para_idx,
                        'source': os.path.basename(pdf_path)
                    })
                    full_text += paragraph + "\n\n"
        
        if not full_text.strip():
            raise ValueError(f"No text could be extracted from {pdf_path}")
        
        return full_text, paragraphs
    
    except Exception as e:
        raise ValueError(f"Failed to read PDF {pdf_path}: {str(e)}")


def chunk_text(
    text: str,
    paragraphs: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    tokenizer_name: str = "cl100k_base"  # OpenAI tokenizer
) -> List[Dict]:
    """
    Split text into chunks with overlap, preserving metadata.
    
    Args:
        text: Full extracted text
        paragraphs: List of paragraph metadata dicts
        chunk_size: Target tokens per chunk (~500-800)
        chunk_overlap: Overlap tokens between chunks (~100)
        tokenizer_name: Tokenizer to use for token counting
    
    Returns:
        List of chunk dicts with text and metadata
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(tokenizer_name)
        
        def token_counter(text: str) -> int:
            return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough character-to-token ratio (1 token ≈ 4 chars)
        def token_counter(text: str) -> int:
            return len(text) // 4
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # Assign metadata to chunks
    chunks_with_metadata = []
    current_paragraph_idx = 0
    
    for chunk_id, chunk in enumerate(chunks):
        # Find which paragraphs this chunk contains
        chunk_start_para = current_paragraph_idx
        chunk_pages = set()
        chunk_paragraphs = set()
        
        # Try to match chunk to paragraphs
        for para in paragraphs[current_paragraph_idx:]:
            if para['text'] in chunk:
                chunk_pages.add(para['page'])
                chunk_paragraphs.add(para['paragraph'])
            if len(chunk_paragraphs) > 0 and para['text'] not in chunk:
                current_paragraph_idx = paragraphs.index(para)
                break
        
        # Use first page and paragraph as reference
        if chunk_pages:
            page = min(chunk_pages)
            paragraph = min(chunk_paragraphs)
        else:
            # Fallback if metadata not found
            page = 1
            paragraph = 1
        
        chunks_with_metadata.append({
            'chunk_id': f"chunk_{chunk_id}",
            'text': chunk,
            'page': page,
            'paragraph': paragraph,
            'source': paragraphs[0]['source'] if paragraphs else "unknown"
        })
    
    return chunks_with_metadata


def get_embeddings():
    """
    Initialize embedding model based on environment configuration.
    
    Returns:
        Embeddings object (OpenAI or HuggingFace)
    """
    use_local = os.getenv('USE_LOCAL_EMBEDDINGS', 'false').lower() == 'true'
    
    if use_local:
        print("Using local embeddings (HuggingFace)...")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    else:
        print("Using OpenAI embeddings...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )


def store_in_chromadb(
    chunks: List[Dict],
    embeddings,
    collection_name: str = "pdf_documents",
    persist_directory: str = "./chromadb_data"
) -> Chroma:
    """
    Store chunks and embeddings in ChromaDB.
    
    Args:
        chunks: List of chunk dicts with text and metadata
        embeddings: Embeddings object
        collection_name: Name of ChromaDB collection
        persist_directory: Path to persist ChromaDB locally
    
    Returns:
        Chroma vector store object
    """
    os.makedirs(persist_directory, exist_ok=True)
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            'chunk_id': chunk['chunk_id'],
            'page': str(chunk['page']),
            'paragraph': str(chunk['paragraph']),
            'source': chunk['source']
        }
        for chunk in chunks
    ]
    
    # Create or update ChromaDB collection
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Persist to disk
    vectorstore.persist()
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_chromadb(
    collection_name: str = "pdf_documents",
    persist_directory: str = "./chromadb_data",
    embeddings=None
) -> Optional[Chroma]:
    """
    Load existing ChromaDB collection.
    
    Args:
        collection_name: Name of ChromaDB collection
        persist_directory: Path to ChromaDB
        embeddings: Embeddings object (required if loading)
    
    Returns:
        Chroma vector store object or None if not found
    """
    if not os.path.exists(persist_directory):
        return None
    
    if embeddings is None:
        embeddings = get_embeddings()
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return vectorstore
    except Exception as e:
        print(f"⚠️ Could not load ChromaDB: {str(e)}")
        return None


def process_pdf(pdf_path: str) -> Tuple[Chroma, List[Dict]]:
    """
    End-to-end PDF processing: load → chunk → embed → store.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (vectorstore, chunks_with_metadata)
    """
    print(f"📄 Loading PDF: {pdf_path}")
    text, paragraphs = load_pdf(pdf_path)
    print(f"✅ Extracted {len(paragraphs)} paragraphs")
    
    print("🔪 Chunking text...")
    chunks = chunk_text(text, paragraphs)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("🤖 Initializing embeddings...")
    embeddings = get_embeddings()
    
    print("💾 Storing in ChromaDB...")
    vectorstore = store_in_chromadb(chunks, embeddings)
    
    return vectorstore, chunks


if __name__ == "__main__":
    # Test with a sample PDF
    test_pdf = "sample.pdf"
    if os.path.exists(test_pdf):
        vectorstore, chunks = process_pdf(test_pdf)
        print(f"\n📊 Sample chunk:\n{chunks[0]}")
    else:
        print(f"❌ Test PDF not found at {test_pdf}")
