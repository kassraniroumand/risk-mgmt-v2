"""
s3 upload utility
--
helper module for centralising all the s3 functionalities
"""
import boto3
import uuid

from src.core.config import settings

# Initialize S3 client using region from settings
s3 = boto3.client("s3", region_name=settings.AWS_REGION)

def upload_pdf_to_s3(file_content: bytes, filename: str) -> tuple[str, str]:
    """Upload a pdf byte to s3 and returns its key and url
    Parameter
    ----
    file_content: bytes
        in-memory pdf payload
    filename:
        original filename from client

    Return:
    ---
    Tuple[str, str]
        (key,url) where key is s3 object path adn url is http url
    """

    # generate unique key based on filename
    key = f"uploads/{uuid.uuid4()}_{filename}"

    # perform the actual upload to s3
    s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=file_content,
        ContentType="application/pdf"
    )

    # construct a https url to uploaded object
    url = f"https://{settings.S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
    return key, url
