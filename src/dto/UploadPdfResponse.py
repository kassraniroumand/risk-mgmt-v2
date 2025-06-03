from pydantic import BaseModel
from typing import Dict, Any

class UploadPdfResponse(BaseModel):
    # input_text: str
    # converted_text: str
    property_valuations_s: Dict[str, Any]
    risk_percentage_s: Dict[str, Any]
    business_interruption_s: Dict[str, Any]
    current_insurance_s: Dict[str, Any]
    multi_currency_risk_s: Dict[str, Any]
    insurance_recommendation_s: Dict[str, Any]
