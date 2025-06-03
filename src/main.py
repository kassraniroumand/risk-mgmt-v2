from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
import os
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as StarletteHTTPException

from exceptions import TextractParseError, S3UploadError, GraphExecutionError
from src.api.endpoint import router
from src.core.config import settings
from src.services.textract_client import parse_pdf_via_textract
from src.utils.s3 import upload_pdf_to_s3
from fastapi import FastAPI, Request

app = FastAPI()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(TextractParseError)
async def textract_exception_handler(request: Request, exc: TextractParseError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "Textract parsing failed",
            "detail": str(exc),
        },
    )


@app.exception_handler(S3UploadError)
async def s3_exception_handler(request: Request, exc: S3UploadError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "S3 upload failed",
            "detail": str(exc),
        },
    )


@app.exception_handler(GraphExecutionError)
async def graph_exception_handler(request: Request, exc: GraphExecutionError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Graph execution failed",
            "detail": str(exc),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Fallback for any other unhandled exception
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
        },
    )


app.include_router(router=router)
