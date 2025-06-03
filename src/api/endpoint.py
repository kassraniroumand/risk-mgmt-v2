"""
Pdf upload and Processing API Module

this module defines routes to
1. Upload PdF files to an aws s3 bucket
2. parse PDF using amazon textract or fetch a cached parse result basd on hash of file from dynamodb to
prevent parsing same document multiple times
4. run langchain graph on parsed text and return a structured resposne
"""
from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import boto3
import time
import re

from exceptions import S3UploadError, TextractParseError, GraphExecutionError, DbExecutionError
from src.dto.UploadPdfResponse import UploadPdfResponse
from src.services.db import hash_text_sha256, get_parsed_text, item_exists, put_parsed_text
from src.services.graph import create_graph
from src.services.textract_client import parse_pdf_via_textract
from src.utils.s3 import upload_pdf_to_s3

router = APIRouter()


@router.get("/")
async def ping():
    """simple health check used by ecs"""
    return {"message": "pong"}


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    """Endpoint to upload pdf file and return structured analysis using DTO

    **WorkFLow**
    1. validate that the uploaded file is a pdf
    2. upload file to s3
    3. extract text using amazon textract or fetch the parsed text from db based on hash of content
    4. run the langchain dag to analyse the pdf using langgraph
    5. return a typed model to client

    Parameter
    ----
    file: uploadFile
        the pdf file send by put request

    Returns
    ----
    UploadPdfResponse
        A pydantic model represent the processed data

    """
    # basic validation
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # read entire file contnet
        contents = await file.read()

        # upload to s3 and return key and url necessary for textract
        try:
            s3_key, s3_url = upload_pdf_to_s3(contents, file.filename)
        except Exception as e:
            raise S3UploadError(f"Failed to upload to S3: {e}")


        text = None
        # get the hash of content for caching
        digest = hash_text_sha256(contents)

        if item_exists(digest):
            # cache hit
            text = get_parsed_text(digest)
        else:
            # cache miss and run textract
            try:
                text = parse_pdf_via_textract(s3_key)
            except Exception as e:
                raise TextractParseError(f"Textract failed on {s3_key}: {e}")
            # persist parsed text into db for caching
            try:
                put_parsed_text(digest, text)
            except  Exception as e:
                raise DbExecutionError(f"Textract failed on {s3_key}: {e}")


        # langchain graph for orchestrating the workflow
        dag = await create_graph()

        # initial state to feed into the graph
        initial_state = {
            "input_text": text,
            "converted_text": "",
            "property_valuations_s": {},
            "risk_percentage_s": {},
            "business_interruption_s": {},
            "current_insurance_s": {},
            "multi_currency_risk_s": {},
            "insurance_recommendation_s": {},
        }

        # execute the DAG async and return final state
        try:
            final_state = await dag.ainvoke(initial_state)
        except Exception as e:
            raise GraphExecutionError(f"Error while running graph: {e}")

        # enforce strict schema
        return UploadPdfResponse(**final_state)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))