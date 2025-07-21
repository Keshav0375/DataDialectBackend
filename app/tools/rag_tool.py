from langchain.tools import tool
from app.utils.chroma_utils import enhanced_similarity_search
from typing import List, Optional


@tool
def rag_search_tool(query: str, document_ids: Optional[List[int]] = None) -> str:
    """
    Enhanced RAG search tool that retrieves relevant context from documents.

    Args:
        query: The search query/question
        document_ids: Optional list of specific document IDs to search within

    Returns:
        Relevant context from documents or error message
    """
    try:
        context = enhanced_similarity_search(
            query=query,
            document_ids=document_ids,
            k=6
        )

        if not context or context.startswith("No relevant"):
            return "I couldn't find relevant information in the uploaded documents for your query."

        return context

    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def document_stats_tool(document_ids: Optional[List[int]] = None) -> str:
    """
    Get statistics about uploaded documents

    Args:
        document_ids: Optional list of document IDs to get stats for

    Returns:
        Document statistics and information
    """
    try:
        from app.utils.db_utils import get_documents_by_ids, get_all_documents

        if document_ids:
            documents = get_documents_by_ids(document_ids)
        else:
            documents = get_all_documents()

        if not documents:
            return "No documents found in the knowledge base."

        stats = []
        for doc in documents:
            stats.append(f"- {doc['filename']} (ID: {doc['id']}, uploaded: {doc['upload_timestamp']})")

        return f"Available documents ({len(documents)} total):\n" + "\n".join(stats)

    except Exception as e:
        return f"Error getting document stats: {str(e)}"