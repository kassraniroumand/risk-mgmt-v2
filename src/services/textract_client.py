"""
Amazon Textract PDF Parsing Utility
---
this helper module provides synchronous and asynchronous wrappers around StartDocumentTextDetection
for extracting text from PDF files already stored
in S3
"""
import asyncio
import time

import boto3
import aioboto3
from src.core.config import settings

textract = boto3.client("textract", region_name=settings.AWS_REGION)

def parse_pdf_via_textract(s3_key: str) -> str:
    """Run Textract document‑text detection and return plain text.

        Parameters
        ----------
        s3_key : str
            Key of the PDF object inside settings.S3_BUCKET
        timeout : int, optional
            Maximum seconds to wait for job completion
        poll_interval : float, optional
            Seconds between GetDocumentTextDetection polls

        Returns
        -------
        str
            Concatenated text detected by Textract
    """
    job = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": settings.S3_BUCKET, "Name": s3_key}}
    )
    # print(job)
    job_id = job["JobId"]

    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        # print(result)
        if result["JobStatus"] in ["SUCCEEDED", "FAILED"]:
            break
        time.sleep(1)

    # print(result)
    lines = [b["Text"] for b in result.get("Blocks", []) if b["BlockType"] == "LINE"]
    text = "\n".join(lines)
    # print(text)
    return text or(f"AWS Textract error: {str(e)}")