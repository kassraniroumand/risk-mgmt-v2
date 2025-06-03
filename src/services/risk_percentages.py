"""
Risk Percentage Extraction Service
---
This module defines two public callables:
    1. build_risk_percentage_prompt: return a parametrised prompt Template
    instructing a llm to identify all probability or
    frequency‑based risk statements
    2. run_risk_percentage: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)


def build_risk_percentage_prompt() -> PromptTemplate:
    prompt_str = r"""
You are a professional insurance risk analyst. Your goal is to scan a plain‐English risk assessment report and extract **all risk factors** that mention a probability, percentage, frequency, or chance. Look for any of these markers:
- A numeric percentage (e.g., “23% probability…”)
- A “X-in-Y-year” phrasing (e.g., “1-in-25-year flood event”)
- A per-timeframe frequency (e.g., “0.3 events per season,” “1.3 postponements per season”)
- A correlation expressed as a percentage (e.g., “15% correlation…”)
- Words like “chance,” “frequency,” “annual probability,” “likelihood,” etc.

Whenever you spot such a phrase, create one JSON object containing:
- **risk_name**: a concise label (often the heading or the short phrase before the number)
- **probability**: the exact numeric expression (e.g., “23%,” “8%,” “1-in-25-year,” “0.3 per season,” “15% correlation”)
- **context**: any qualifier of time or scope (e.g., “annually,” “per season,” “during winter,” etc.)
- **notes**: any additional qualifiers (e.g., cost ranges, priority labels, correlation notes)

**IMPORTANT**: You must output all currency values in **EUR**, even if the input contains GBP or USD.

Here is the extracted report text:

{cleaned_text}

---

Below are two examples illustrating the expected format:

**Example 1**  
_Excerpt_:  
> “Weather-Related Risks: There is a 23% probability of weather-related damage requiring emergency repairs annually.”  
_Output_:  
```json
{{
  "risk_name":   "Weather-Related Risk",
  "probability": "23%",
  "context":     "annually",
  "notes":       ""
}}
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)


async def run_risk_percentage(state: dict) -> dict:
    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_risk_percentage_prompt()
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
        return {"risk_percentage_s": wrapped}
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise
