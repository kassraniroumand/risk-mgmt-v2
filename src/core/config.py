"""
Runtime config module
----
centralises the retrieval of runtime configuration parameters from aws parameter store,
***explain in report pdf about infra
"""
import boto3

ssm = boto3.client("ssm", region_name="us-east-1")  # static or dynamic region

def get_parameter(name: str, with_decryption: bool = True) -> str:
    """retrieve a parameter from ssm
    Parameter
    ---
    name: str
        fully qualified parameter path
    with_decryption:
        if true, kms encrypted secure string params
    """
    return ssm.get_parameter(Name=name, WithDecryption=with_decryption)["Parameter"]["Value"]

class Settings:
    """load critical run-time settings from ssm on instantiation"""
    def __init__(self):
        # api key for groq
        self.GROQ_API_KEY = get_parameter("/ai-reporter/prod/groq_api_key")
        # s3 config
        self.S3_BUCKET = get_parameter("/ai-reporter/prod/s3_bucket")
        # aws region
        self.AWS_REGION = get_parameter("/ai-reporter/prod/aws_region")
        # pre fetch the rate
        self.EXCHANGE_RATES = {
            "EUR": float(get_parameter("/ai-reporter/prod/exchange_rate_eur")),
            "USD": float(get_parameter("/ai-reporter/prod/exchange_rate_usd")),
            "GBP": float(get_parameter("/ai-reporter/prod/exchange_rate_gbp")),
        }

# module level singleton like singleton pattern
settings = Settings()
