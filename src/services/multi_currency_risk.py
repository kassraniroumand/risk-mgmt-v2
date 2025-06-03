"""
Multi-Currency Risk Extraction Service
---
    1. build_multi_currency_risk_prompt: return a parametrised prompt Template
    instructing an llm to identifying foreign exchange and multi‑currency
    risk
    2. run_multy_currency_risk: LangGraph compatible async node that feeds
    converted_text through the prompt
"""
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)

def build_multi_currency_risk_prompt() -> PromptTemplate:
    prompt_str = r"""
You are a professional treasury-risk analyst.  
Find every multi-currency risk factor in the text that contains a percentage, probability, frequency, or quantified FX exposure.

Return **one JSON object** whose **top-level keys are the risk names** and whose values are dictionaries with exactly these keys:
- probability
- context
- notes  (convert GBP→EUR at 1 GBP = 1.17 EUR and USD→EUR at 1 USD = 0.92 EUR)

Example output:
```json
{{
  "EUR/GBP Exchange Risk": {{
    "probability": "12%",
    "context": "annual costs under FX fluctuation",
    "notes": ""
  }}
}}
Here is the report excerpt to analyse:

{cleaned_text}
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)

async def run_multy_currency_risk(state: dict) -> dict:
    """LangGraph node to extract multi‑currency risks and merge into state."""
    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_multi_currency_risk_prompt()
    parser = JsonOutputParser()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    chain: Runnable = prompt | llm | parser

    try:
        result = await chain.ainvoke({"cleaned_text": cleaned_text})
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise

    return {"multi_currency_risk_s": result}
