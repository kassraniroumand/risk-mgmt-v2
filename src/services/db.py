"""
DynamoDB Persistence and Utility functions
----
this module centralises DynamoDB read/write helpers and generic sha-256 hash function used
in application. all interactions target the `parsedText` table

Key Responsibility
---
get_parsed_text: fetch previously parsed text by its textId
put_parsed_text: persist a new textID
item_exists: constant cost existence check
hash_text_sha256: asset agnostic hashing
"""
import hashlib
from typing import Union

import boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  # adjust region as needed
table = dynamodb.Table('parseText')

def get_parsed_text(text_id):
    """
    return the *parsed text* for a given textId or none if absent

    Parameter
    ---
    text_id: str
        partition key of the dynamoDB item

    Return
    ---
    optional str
        the parseText of the dynamoDb item
    """
    try:
        response = table.get_item(
            Key={
                'textID': text_id
            }
        )
    except ClientError as e:
        print(f"Unable to fetch item: {e.response['Error']['Message']}")
        return None

    # The returned dict will contain 'Item' if the key was found
    item = response.get('Item')
    if not item:
        print(f"No item found with textID={text_id}")
        return None

    # Return the parseText attribute
    return item.get('parseText')

def put_parsed_text(text_id, parsed_text):
    """
    Inserts an item into DynamoDB with partition key textID
    and attribute parseText.
    """
    try:
        response = table.put_item(
            Item={
                'textID': text_id,        # Partition key
                'parseText': parsed_text  # Additional attribute
                # You can add more attributes here as needed
            }
        )
        print(f"Item inserted (textID={text_id})")
        return response
    except ClientError as e:
        print(f"Failed to insert item: {e.response['Error']['Message']}")


def item_exists(text_id: str) -> bool:
    """
    Returns True if an item with partition key 'textID' == text_id exists in the table;
    otherwise returns False.
    """
    try:
        response = table.get_item(
            Key={'textID': text_id},
            ProjectionExpression='textID'  # only fetch the key itself to minimize read cost
        )
    except ClientError as e:
        # Log the error or re‐raise as needed
        print(f"Unable to check existence: {e.response['Error']['Message']}")
        return False

    # If the key was found, 'Item' will be present in the response
    return 'Item' in response

#%%

def hash_text_sha256(data: Union[str, bytes]) -> str:
    """
    Compute SHA-256 of either a `str` or `bytes` input, returning a hex string

    If data is a str, it’s UTF 8encoded under the hood
    If data is already bytes, it’s passed directly to the hasher.
    """
    # 1. Normalize to bytes
    if isinstance(data, str):
        to_hash = data.encode('utf-8')
    elif isinstance(data, (bytes, bytearray)):
        to_hash = bytes(data)
    else:
        raise TypeError("hash_text_sha256(): expected str or bytes, got %r" % type(data))

    # 2. Create SHA-256 hasher and feed in the bytes
    sha256 = hashlib.sha256()
    sha256.update(to_hash)
    return sha256.hexdigest()
