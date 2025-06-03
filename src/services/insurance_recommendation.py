"""
Insurance Recommendation Extraction Service
----
This module exposes two callables:
    1. build_insurance_recommendation_prompt: return a parametrised prompt Template
    instructing an llm to  pull structured recommendations
    2. run_insurance_recommendation: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)

def build_insurance_recommendation_prompt() -> PromptTemplate:
    prompt_str = r"""
You are an expert insurance analyst.

TASK  
From the text block provided {cleaned_text}, identify every *distinct insurance recommendation* and extract the following four fields for each:

1. **Coverage & Structure** – a short label that states the type of cover, limit, and any key structural features (e.g., “Parametric weather cover – £1 m per trigger”).
2. **Business Rationale** – one or two sentences explaining *why* the cover is needed (drivers such as severity, probability, regulatory needs, etc.).
3. **Implementation Timeline / Priority** – the recommended urgency plus any specific dates, milestones, or season markers mentioned.
4. **Financial Impact** – the estimated premium change or cost saving (give currency and amount, or state “Not specified” if none).

OUTPUT
Return **only** a JSON object whose keys are the `bi_name` values (slug-or-snake-cased), and whose values are dictionaries with the other four fields.

- "coverage"
- "rationale"
- "timeline"
- "financial_impact"

EXAMPLE OUTPUT  
```json
  {{
    "coverage": "Increase Property & BI limit to £25 m CSL (12-month indemnity)",
    "rationale": "Captures worst-case loss of €18 m plus 25 % buffer for FX volatility.",
    "timeline": "High priority – bind before August 2025 season opener; market submissions by 15 Jun 2025.",
    "financial_impact": "Premium uplift ~£220k"
  }},
  …
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)

async def run_insurance_recommendation(state: dict) -> dict:
    """LangGraph node to extract insurance recommendations and merge into state"""
    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_insurance_recommendation_prompt()
    parser = JsonOutputParser()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    chain: Runnable = prompt | llm | parser

    try:
        result = await chain.ainvoke({"cleaned_text": cleaned_text})
        if isinstance(result, list):
            wrapped = {"risks": result}
        else:
            wrapped = result
        return {"insurance_recommendation_s": result}
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise

