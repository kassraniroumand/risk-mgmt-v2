"""
Business Interruption Extraction Service
----
This module defines two public callables:
    1. build_business_interruption_prompt: return a parametrised prompt Template
    instructing an llm to identify and normalise Business Interruption
    2. run_business_interruption: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
import logging

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from src.core.config import settings

logger = logging.getLogger(__name__)

def build_business_interruption_prompt() -> PromptTemplate:
    """Construct the BI‑extraction prompt

       The prompt:
        Explains the analyst's goal
        Provides marker phrases to look for
        Embeds the report text via cleaned_text
        Specifies a JSON schema for the output
       """
    prompt_str = r"""
You are a professional insurance BI analyst.

GOAL  
Scan the plain-English risk-assessment report below and extract **every statement that quantifies a Business Interruption (BI) exposure**—i.e., any figure representing lost revenue, extra expense, or cost impact arising from disrupted football operations (matches, training, maintenance overruns, alternate venues, etc.).

MARKERS to watch for  
- Phrases containing **“lost revenue,” “revenue impact,” “business interruption,” “closure,” “postponement,” “disruption,” “alternate-venue costs,” “training facility costs,”** etc.  
- Any money amount linked to a timeframe (per match, per week, per season, total during closure, etc.).  
- Words indicating BI magnitude even if the term “business interruption” isn’t used explicitly.

---

Here is the extracted report text:

{cleaned_text}
----------------

OUTPUT  
For each BI figure you find, create **one entry** in a JSON object whose keys are the *BI labels* and whose values are dictionaries with exactly these fields:

| Field          | Description                                                                                           |
|----------------|-------------------------------------------------------------------------------------------------------|
| **"amount_eur"** | Monetary value **in EUR**, numeric only (no commas, symbols)                                         |
| **"timeframe"**  | Period the amount refers to (e.g., “per home fixture,” “10-week closure,” “annually”)               |
| **"quote"**      | Verbatim sentence(s) from the document that state the amount, preserving punctuation                 |
| **"notes"**      | Optional qualifiers (trigger details, conversion notes). Convert all GBP and USD at €1 = £0.85 and €1 = $1 / 1.18, rounding to the nearest euro |

Return **only** the resulting JSON object—no wrapper keys beyond the BI labels themselves, no headings, and no commentary.

---

### Example output  
_Excerpt (illustration only)_  
> “Each Premier League fixture generates approximately €8.5 million EUR in combined revenue streams.”

```json
{{
  "Match revenue per fixture": {{
    "amount_eur": 8500000,
    "timeframe": "per home fixture",
    "quote": "Each Premier League fixture generates approximately €8.5 million EUR in combined revenue streams.",
    "notes": ""
  }}
}}
"""
    return PromptTemplate(input_variables=["cleaned_text"], template=prompt_str)

async def run_business_interruption(state: dict) -> dict:
    """LangGraph node to extract BI exposures.

        Parameters
        ----------
        state : dict
            Shared graph state; must contain key converted_text.

        Returns
        -------
        dict
            New state fragment `{ "business_interruption_s": <JSON object> }`.
        """

    logger.info("Current state at logger_node: %s", state)
    cleaned_text = state.get("converted_text")
    if not cleaned_text:
        logger.error("Missing 'converted_text' in state")
        raise ValueError("Missing 'converted_text' key in state dict")

    prompt = build_business_interruption_prompt()
    parser = JsonOutputParser()

    # LLM configuration
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    # Compose LangChain Runnable
    chain: Runnable = prompt | llm | parser

    try:
        # Asynchronously invoke the node execution
        result = await chain.ainvoke({"cleaned_text": cleaned_text})
        return {"business_interruption_s": result}
    except Exception as e:
        logger.error("Chain invocation failed: %s", e)
        raise