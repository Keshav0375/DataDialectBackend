from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None

    # Database Configuration
    mysql_host: str = "localhost"
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "classicmodels"

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "chat_db"

    # Performance Settings
    response_timeout: int = 30
    max_concurrent_requests: int = 50

    # Chat Configuration
    default_temperature: float = 0.7
    max_tokens: int = 1000

    class Config:
        env_file = ".env"


settings = Settings()