"""
Property Valuation & Executive Summary Extraction Service
---
This module defines two public callables:
    1. build_insurance_analysis_prompt: return a parametrised prompt Template
    instructing an llm to derive an executive summary of key
   financial metrics
    2. run_property_valuation: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)

def build_insurance_analysis_prompt() -> PromptTemplate:
    prompt_str = """
You are a professional insurance risk analyst with experience in commercial property, public liability, and business interruption policies.

The following text was extracted from a risk assessment report for a major football stadium. Your tasks are:

---

**1. Extract Key Risk and Financial Data**
- Summarize:
  - Total property valuation (across playing surface and infrastructure)
  - Business interruption exposure (e.g., match cancellations, lost revenue)
  - Annual operating costs
  - Historical and forecasted damage costs

---

**IMPORTANT**: You must output all currency values in **EUR**, even if the input contains GBP or USD.

Here is the extracted report text:

{cleaned_text}

---

**Output Format:**  
Respond *only* with a single JSON object as shown below, without any additional text, headers, or explanations:

{{
  "executive_summary": "..."
}}
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)

async def run_property_valuation(state: dict) -> dict:
    """LangGraph node to generate an executive summary and merge into state."""
    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_insurance_analysis_prompt()
    parser = JsonOutputParser()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    chain: Runnable = prompt | llm | parser

    try:
        result = await chain.ainvoke({"cleaned_text": cleaned_text})
        return {"property_valuations_s": result}
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise