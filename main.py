from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.credentials import router as credentials_router
from app.api.endpoints.csv_upload import router as csv_router
from app.api.endpoints.python_upload import router as python_router
from app.api.endpoints.status import router as status_router
from app.api.endpoints.sql_query import router as sql_query
from app.api.endpoints.no_sql_query import router as nosql_query
from app.api.endpoints.schema_generator import router as schema_generator

app = FastAPI(
    title="DataDialect",
    description="Multi-Database High-performance chat system with SQL, NoSQL, and Document databases",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(credentials_router, prefix="/api/v1", tags=["credentials"])
app.include_router(csv_router, prefix="/api/v1", tags=["csv-upload"])
app.include_router(python_router, prefix="/api/v1", tags=["python-upload"])
app.include_router(status_router, prefix="/api/v1", tags=["status"])
app.include_router(sql_query, prefix="/api/v1", tags=["sql-query"])
app.include_router(schema_generator, prefix="/api/v1", tags=["schema-generator"])
app.include_router(nosql_query, prefix="/api/v1", tags=["nosql-query"])


@app.get("/")
async def root():
    return {"message": "DataDialect API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "DataDialect-multi-db-chatbot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)