from fastapi import FastAPI

from app.api.endpoints.credentials import router as credentials_router
from app.api.endpoints.csv_upload import router as csv_router
from app.api.endpoints.python_upload import router as python_router
from app.api.endpoints.status import router as status_router

app = FastAPI(
    title="DataDialect",
    description="Multi-Database High-performance chat system with SQL, NoSQL, and Document databases",
    version="1.0.0"
)

app.include_router(credentials_router, prefix="/api/v1", tags=["credentials"])
app.include_router(csv_router, prefix="/api/v1", tags=["csv-upload"])
app.include_router(python_router, prefix="/api/v1", tags=["python-upload"])
app.include_router(status_router, prefix="/api/v1", tags=["status"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "DataDialect-multi-db-chatbot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)