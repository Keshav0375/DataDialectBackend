import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

DB_NAME = "rag_app.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_chat_history():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


def insert_chat_history(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO chat_history (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM chat_history WHERE session_id = ? ORDER BY created_at',
                   (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages


def create_document_store():
    """Enhanced document store with collection support"""
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT NOT NULL,
                     collection_id TEXT,
                     file_size INTEGER,
                     file_type TEXT,
                     chunk_count INTEGER DEFAULT 0,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     metadata TEXT)''')

    conn.execute('''CREATE INDEX IF NOT EXISTS idx_collection_id 
                    ON document_store(collection_id)''')
    conn.close()


def insert_document_record(filename: str, collection_id: str = None, file_size: int = None,
                           file_type: str = None, metadata: Dict[str, Any] = None) -> int:
    """Enhanced document record insertion"""
    conn = get_db_connection()
    cursor = conn.cursor()

    if not file_type and '.' in filename:
        file_type = filename.split('.')[-1].lower()

    metadata_json = json.dumps(metadata) if metadata else None

    cursor.execute('''INSERT INTO document_store 
                      (filename, collection_id, file_size, file_type, metadata) 
                      VALUES (?, ?, ?, ?, ?)''',
                   (filename, collection_id, file_size, file_type, metadata_json))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def update_document_chunk_count(file_id: int, chunk_count: int):
    """Update the chunk count for a document"""
    conn = get_db_connection()
    conn.execute('UPDATE document_store SET chunk_count = ? WHERE id = ?',
                 (chunk_count, file_id))
    conn.commit()
    conn.close()


def delete_document_record(file_id: int) -> bool:
    """Delete a document record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    affected_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return affected_rows > 0


def get_all_documents() -> List[Dict[str, Any]]:
    """Get all documents with enhanced information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT id, filename, collection_id, file_size, file_type, 
                             chunk_count, upload_timestamp, metadata 
                      FROM document_store ORDER BY upload_timestamp DESC''')
    documents = []
    for row in cursor.fetchall():
        doc = dict(row)
        if doc['metadata']:
            try:
                doc['metadata'] = json.loads(doc['metadata'])
            except:
                doc['metadata'] = {}
        documents.append(doc)
    conn.close()
    return documents


def get_documents_by_ids(document_ids: List[int]) -> List[Dict[str, Any]]:
    """Get documents by their IDs"""
    if not document_ids:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    placeholders = ','.join(['?' for _ in document_ids])
    cursor.execute(f'''SELECT id, filename, collection_id, file_size, file_type, 
                              chunk_count, upload_timestamp, metadata 
                       FROM document_store 
                       WHERE id IN ({placeholders})
                       ORDER BY upload_timestamp DESC''', document_ids)

    documents = []
    for row in cursor.fetchall():
        doc = dict(row)
        if doc['metadata']:
            try:
                doc['metadata'] = json.loads(doc['metadata'])
            except:
                doc['metadata'] = {}
        documents.append(doc)
    conn.close()
    return documents


def get_documents_by_collection_id(collection_id: str) -> List[Dict[str, Any]]:
    """Get all documents in a specific collection"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT id, filename, collection_id, file_size, file_type, 
                             chunk_count, upload_timestamp, metadata 
                      FROM document_store 
                      WHERE collection_id = ? 
                      ORDER BY upload_timestamp DESC''', (collection_id,))

    documents = []
    for row in cursor.fetchall():
        doc = dict(row)
        if doc['metadata']:
            try:
                doc['metadata'] = json.loads(doc['metadata'])
            except:
                doc['metadata'] = {}
        documents.append(doc)
    conn.close()
    return documents


def get_collection_stats(collection_id: str) -> Dict[str, Any]:
    """Get statistics for a document collection"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''SELECT COUNT(*) as doc_count, 
                             SUM(file_size) as total_size,
                             SUM(chunk_count) as total_chunks,
                             MIN(upload_timestamp) as first_upload,
                             MAX(upload_timestamp) as last_upload
                      FROM document_store 
                      WHERE collection_id = ?''', (collection_id,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "collection_id": collection_id,
            "document_count": result['doc_count'],
            "total_size": result['total_size'] or 0,
            "total_chunks": result['total_chunks'] or 0,
            "first_upload": result['first_upload'],
            "last_upload": result['last_upload']
        }
    else:
        return {
            "collection_id": collection_id,
            "document_count": 0,
            "total_size": 0,
            "total_chunks": 0,
            "first_upload": None,
            "last_upload": None
        }


def delete_collection_documents(collection_id: str) -> int:
    """Delete all documents in a collection and return count of deleted documents"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM document_store WHERE collection_id = ?', (collection_id,))
    affected_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return affected_rows


def search_documents(query: str, collection_id: str = None) -> List[Dict[str, Any]]:
    """Search documents by filename or metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()

    if collection_id:
        cursor.execute('''SELECT id, filename, collection_id, file_size, file_type, 
                                 chunk_count, upload_timestamp, metadata 
                          FROM document_store 
                          WHERE collection_id = ? AND filename LIKE ?
                          ORDER BY upload_timestamp DESC''',
                       (collection_id, f'%{query}%'))
    else:
        cursor.execute('''SELECT id, filename, collection_id, file_size, file_type, 
                                 chunk_count, upload_timestamp, metadata 
                          FROM document_store 
                          WHERE filename LIKE ?
                          ORDER BY upload_timestamp DESC''',
                       (f'%{query}%',))

    documents = []
    for row in cursor.fetchall():
        doc = dict(row)
        if doc['metadata']:
            try:
                doc['metadata'] = json.loads(doc['metadata'])
            except:
                doc['metadata'] = {}
        documents.append(doc)
    conn.close()
    return documents


create_chat_history()
create_document_store()