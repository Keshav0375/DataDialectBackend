import os
from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models.schemas import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from app.utils.db_utils import insert_chat_history, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record, get_documents_by_ids
from app.utils.chroma_utils import index_document_to_chroma, delete_doc_from_chroma, create_vector_store_collection
from app.nodes.RAG import rag_graph_instance  # Import the global instance
from langchain_core.messages import HumanMessage, AIMessage
from logging_config import setup_logger
import shutil
from app.utils.basic_utils import get_or_create_session_id, history_to_lc_messages, append_message
from app.utils.rag_langchain_utils import contextualise_chain
from typing import List, Optional
import tempfile


logger = setup_logger("DataDialect", "DataDialect.log")
router = APIRouter()

load_dotenv()


@router.post("/rag-chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """
    Main chat endpoint using the LangGraph agent with routing,
    RAG, web search capabilities.
    """

    session_id = get_or_create_session_id(query_input.session_id)
    logger.info(
        f"Session ID: {session_id}, User Query: {query_input.question}, Document IDs: {query_input.document_ids}")

    try:
        # Validate document IDs if provided
        if query_input.document_ids:
            documents = get_documents_by_ids(query_input.document_ids)
            if len(documents) != len(query_input.document_ids):
                raise HTTPException(status_code=404, detail="Some document IDs not found")
            logger.info(f"Using documents: {[doc['filename'] for doc in documents]}")

        chat_history = get_chat_history(session_id)
        messages = history_to_lc_messages(chat_history)

        standalone_q = contextualise_chain.invoke({
            "chat_history": messages,
            "input": query_input.question
        })

        messages = append_message(messages, HumanMessage(content=standalone_q))

        initial_state = {
            "messages": messages,
            "route": None,
            "rag": None,
            "web": None,
            "document_ids": query_input.document_ids or []
        }

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        result = rag_graph_instance.invoke(initial_state, config=config)

        if not result.get("success", True):
            logger.error(f"RAG workflow failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=f"RAG workflow error: {result.get('error', 'Unknown error')}")

        last_message = None
        if result.get("messages"):
            last_message = next((m for m in reversed(result["messages"])
                                 if isinstance(m, AIMessage)), None)

        if last_message:
            answer = last_message.content
        else:
            answer = "I apologize, but I couldn't generate a response at this time"

        insert_chat_history(session_id, query_input.question, answer, getattr(query_input, 'model', 'rag'))
        logger.info(f"Session ID: {session_id}, AI Response: {answer}")

        return QueryResponse(
            answer=answer,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.post("/upload-doc")
def upload_and_index_document(files: List[UploadFile] = File(...)):
    """Upload and index a document to the knowledge base"""
    allowed_extensions = ['.pdf', '.docx', '.txt']

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for file in files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.filename}'. Allowed types are: {', '.join(allowed_extensions)}"
            )

    uploaded_documents = []
    temp_file_paths = []

    try:
        collection_id = create_vector_store_collection()
        logger.info(f"Created collection ID: {collection_id}")

        for file in files:
            temp_file_path = f"temp_{collection_id}_{file.filename}"
            temp_file_paths.append(temp_file_path)

            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_id = insert_document_record(file.filename, collection_id)

            success = index_document_to_chroma(temp_file_path, file_id, collection_id)

            if success:
                uploaded_documents.append({
                    "filename": file.filename,
                    "file_id": file_id,
                    "size": file.size if hasattr(file, 'size') else 0
                })
                logger.info(f"Successfully indexed {file.filename} with file_id: {file_id}")
            else:
                delete_document_record(file_id)
                raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}")

        return {
                "success": True,
                "message": f"Successfully uploaded and indexed {len(uploaded_documents)} document(s)",
                "collection_id": collection_id,
                "total_files": len(uploaded_documents),
                "documents": uploaded_documents,  # Contains file_ids for each document
                "file_ids": [doc["file_id"] for doc in uploaded_documents]  # Easy access to all file IDs
            }
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        for doc in uploaded_documents:
            try:
                delete_document_record(doc["file_id"])
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    finally:
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

@router.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    """List all documents in the knowledge base"""
    try:
        return get_all_documents()
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.post("/delete-docs")
def delete_documents(request: List[int]):
    """Delete multiple documents from the knowledge base"""
    try:
        success_count = 0
        failed_files = []

        for file_id in request:
            try:
                # First delete from Chroma vector store
                chroma_delete_success = delete_doc_from_chroma(file_id)

                if chroma_delete_success:
                    # Then delete from database
                    db_delete_success = delete_document_record(file_id)
                    if db_delete_success:
                        success_count += 1
                    else:
                        failed_files.append(f"file_id {file_id} (database deletion failed)")
                else:
                    failed_files.append(f"file_id {file_id} (vector store deletion failed)")
            except Exception as e:
                failed_files.append(f"file_id {file_id} (error: {str(e)})")

        if failed_files:
            return {
                "message": f"Partially successful: {success_count} deleted, {len(failed_files)} failed",
                "success_count": success_count,
                "failed_files": failed_files
            }
        else:
            return {
                "message": f"Successfully deleted {success_count} document(s)",
                "success_count": success_count
            }

    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")

