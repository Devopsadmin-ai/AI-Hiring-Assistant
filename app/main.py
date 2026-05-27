from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.v1 import router as v1_router
from fastapi.responses import JSONResponse
from app.core.logger import setup_logger
from app.core.config import settings

logger = setup_logger(__name__)

app = FastAPI(
    title="AI Hiring Assistant",
    description=f"LLM : `{settings.LLM_PROVIDER}` | Model : `{settings.GEMINI_MODEL}`",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(v1_router)


def _err(code: str, message: str) -> dict:
    return {
        "error": code,
        "message": message
    }


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.error(f"[422] {request.method} {request.url} | {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=_err("UNPROCESSABLE", "Input could not be parsed or analysed.")
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        logger.error(f"[{exc.status_code}] {request.method} {request.url}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )

    if exc.status_code == 404:
        logger.error(f"[404] {request.method} {request.url}")
        return JSONResponse(
            status_code=404,
            content=_err("NOT_FOUND", "Resource not found.")
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=_err("HTTP_ERROR", str(exc.detail))
    )


@app.exception_handler(Exception)
async def server_error_handler(request: Request, exc: Exception):
    logger.error(f"[500] {request.method} {request.url} | {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
    )


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.GEMINI_MODEL
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AI Hiring Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/jd/analyze",
            "POST /api/v1/resume/analyze",
            "POST /api/v1/resume/analyze/batch",
            "POST /api/v1/interview/plan",
            "POST /api/v1/risk/analyze",
            "POST /api/v1/interview/analytical"
        ]
    }
