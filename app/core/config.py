from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Primary LLM Configuration
    LLM_PROVIDER: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = ""

    # Secondary LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = ""

    # Frontend Configuration
    FRONTEND_API_BASE: str = ""
    FRONTEND_API_TOKEN: str = ""

    # Backend Configuration
    BACKEND_API_KEY: str = ""

    # Apify Configuration
    APIFY_API_KEY: str = ""
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
