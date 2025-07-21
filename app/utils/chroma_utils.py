from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional
import os
import uuid
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import re

load_dotenv()

azure_api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION", "2023-05-15")
openai_api_key = os.getenv("OPENAI_API_KEY")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False
)

large_doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False
)

embedding_function = AzureOpenAIEmbeddings(
    azure_deployment="Text-Analytics",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def create_vector_store_collection() -> str:
    """Create a unique collection ID for a batch of documents"""
    return str(uuid.uuid4())


def load_and_split_document(file_path: str) -> List[Document]:
    """Enhanced document loading with better text processing"""

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    try:
        documents = loader.load()

        for doc in documents:
            doc.page_content = clean_text_content(doc.page_content)

        total_content = "\n".join([doc.page_content for doc in documents])

        if len(total_content) > 50000:
            splits = large_doc_splitter.split_documents(documents)
        else:
            splits = text_splitter.split_documents(documents)

        for i, split in enumerate(splits):
            split.metadata.update({
                'chunk_index': i,
                'total_chunks': len(splits),
                'file_type': file_path.split('.')[-1],
                'chunk_size': len(split.page_content)
            })

        return splits


    # Replace the current exception handling

    except Exception as e:

        print(f"Error loading document {file_path}: {str(e)}")

        raise Exception(f"Error loading document {file_path}: {str(e)}")


def clean_text_content(text: str) -> str:
    """Clean and normalize text content for better processing"""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]

    return '\n'.join(cleaned_lines).strip()


def index_document_to_chroma(file_path: str, file_id: int, collection_id: str) -> bool:
    """Enhanced document indexing with better metadata"""
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata.update({
                'file_id': file_id,
                'collection_id': collection_id,
                'source_file': os.path.basename(file_path),
                'indexed_at': str(uuid.uuid4())  # Unique ID for each chunk
            })

        vectorstore.add_documents(splits)

        print(f"Successfully indexed {len(splits)} chunks from {file_path}")
        return True

    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def enhanced_similarity_search(query: str, document_ids: Optional[List[int]] = None, k: int = 6) -> str:
    """
    Enhanced similarity search with better context retrieval
    """
    try:
        filter_dict = {}
        if document_ids:
            filter_dict["file_id"] = {"$in": document_ids}

        if filter_dict:
            docs = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            docs = vectorstore.similarity_search(query, k=k)

        if not docs:
            return "No relevant documents found in the knowledge base."

        context_parts = []
        seen_content = set()
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            if content in seen_content or len(content) < 20:
                continue
            seen_content.add(content)

            source_info = f"Source: {doc.metadata.get('source_file', 'Unknown')}"
            chunk_info = f"Chunk {doc.metadata.get('chunk_index', i) + 1}"

            formatted_chunk = f"[{source_info} - {chunk_info}]\n{content}\n"
            context_parts.append(formatted_chunk)

        if not context_parts:
            return "No relevant content found in the documents."

        context = "\n" + "=" * 50 + "\n".join(context_parts) + "=" * 50 + "\n"

        print(f"Retrieved {len(context_parts)} relevant chunks from documents")
        return context

    except Exception as e:
        print(f"Error in similarity search: {e}")
        return f"Error searching documents: {str(e)}"


def delete_doc_from_chroma(file_id: int) -> bool:
    """Enhanced document deletion"""
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        if docs['ids']:
            vectorstore.delete(ids=docs['ids'])
            print(f"Deleted {len(docs['ids'])} document chunks with file_id {file_id}")
        else:
            print(f"No document chunks found with file_id {file_id}")
        return True

    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False


def delete_collection_from_chroma(collection_id: str) -> bool:
    """Delete all documents in a collection"""
    try:
        docs = vectorstore.get(where={"collection_id": collection_id})
        print(f"Found {len(docs['ids'])} document chunks for collection_id {collection_id}")

        if docs['ids']:
            vectorstore.delete(ids=docs['ids'])
            print(f"Deleted {len(docs['ids'])} document chunks with collection_id {collection_id}")
        else:
            print(f"No document chunks found with collection_id {collection_id}")
        return True

    except Exception as e:
        print(f"Error deleting collection with collection_id {collection_id} from Chroma: {str(e)}")
        return False


def get_collection_stats(collection_id: str) -> dict:
    """Get statistics about a document collection"""
    try:
        docs = vectorstore.get(where={"collection_id": collection_id})

        if not docs['ids']:
            return {"total_chunks": 0, "files": []}

        file_stats = {}
        for metadata in docs['metadatas']:
            file_id = metadata.get('file_id')
            source_file = metadata.get('source_file', 'Unknown')

            if file_id not in file_stats:
                file_stats[file_id] = {
                    "file_id": file_id,
                    "source_file": source_file,
                    "chunk_count": 0
                }
            file_stats[file_id]["chunk_count"] += 1

        return {
            "collection_id": collection_id,
            "total_chunks": len(docs['ids']),
            "file_count": len(file_stats),
            "files": list(file_stats.values())
        }

    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {"error": str(e)}