"""
Current Insurance Gap Extraction Service
----
This module exposes two public callables:
    1. build_current_insurance_prompt: return a parametrised prompt Template
    instructing an llm to identify coverage gaps in the
    existing insurance programme
    2. run_current_insurance: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)

def build_current_insurance_prompt() -> PromptTemplate:
    """Construct the coverage‑gap extraction prompt."""

    prompt_str = """
You are a professional insurance coverage analyst.

GOAL  
Scan the plain-English risk-assessment report below and extract **every statement that indicates an inadequacy, exclusion, or gap in the EXISTING insurance programme**.

MARKERS to watch for  
- Words or phrases such as **“insufficient,” “inadequate,” “does not account,” “excludes,” “lacks,” “gap,” “mismatch,” “not addressed,”** “currently lacks,” “coverage limits appear…,” etc.  
- Mentions of specific policy sections or limits that fall short of replacement costs, business-interruption losses, liability exposures, currency risks, specialist-contractor needs, or regulatory requirements.

OUTPUT  
Whenever you spot such a gap, create **one** JSON object with **exactly** these keys:

- **"gap_name"** → a concise (≤ 8-word) label for the gap  
- **"issue"**  → a short phrase capturing what’s missing or deficient (e.g., “Property limits too low,” “No parametric weather cover”)  
- **"quote"**  → the verbatim sentence(s) from the document that prove the gap, preserving punctuation and ellipses (no paraphrasing)  
- **"notes"**  → any extra nuance—e.g., cost ranges, currencies, loss estimates, or who is affected—converted to EUR where figures are given (assume €1 = £0.85 and €1 = $1/1.18 if needed)

Wrap **all** gap objects inside a single top-level key named **"current_insurance_gaps"** and return **only the JSON**—no headings or commentary.

---

Here is the extracted report text:

{cleaned_text}
----------------

### Example output  
_Excerpt (for illustration only)_:  
> “Current property limits appear insufficient when considering the full replacement timeline … particularly given the €12.8 million EUR revenue impact during closure periods.”

```json
{{
      "gap_name": "Property limits vs replacement",
      "issue": "Property limits too low",
      "quote": "Current property limits appear insufficient when considering the full replacement timeline … particularly given the €12.8 million EUR revenue impact during closure periods.",
      "notes": ""
}}
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)

async def run_current_insurance(state: dict) -> dict:
    """LangGraph node to extract current insurance gaps"""
    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_current_insurance_prompt()
    parser = JsonOutputParser()
    print("GROQ_API_KEY", settings.GROQ_API_KEY)
    # Instantiate Groq LLM once per invocation
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    chain: Runnable = prompt | llm | parser

    try:
        # Asynchronously invoke the node execution
        result = await chain.ainvoke({"cleaned_text": cleaned_text})
        return {"current_insurance_s": result}
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise

