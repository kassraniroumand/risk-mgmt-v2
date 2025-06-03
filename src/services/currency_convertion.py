"""
Currency Conversion & Normalisation Service
---
This module defines two public callables:
    1. build_currency_conversion_prompt: return a parametrised prompt Template
    instructing an llm to detect & convert every monetary amount in a text block
   from USD/GBP/EUR
    2. run_currency_conversion: LangGraph compatible async node that feeds
   converted_text through the prompt
"""
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable
from src.core.config import settings


def build_currency_conversion_prompt() -> PromptTemplate:
    prompt_str = """
You are a financial assistant. The following text contains monetary values in USD ($), GBP (£), and EUR (€).

Your task:
1. Detect all monetary values, including those with multipliers such as "million", "m", "thousand", or "k".
2. Convert each amount to EUR using the following fixed rates:
   - 1 USD = 0.91 EUR
   - 1 GBP = 1.18 EUR
3. Multiply values accordingly before conversion:
   - "million" or "m" = ×1,000,000
   - "thousand" or "k" = ×1,000

Formatting Rules:
- Round every converted amount to 2 decimal places.
- Format with comma as thousand separator and dot as decimal (e.g., `1,200.50 EUR`).
- Keep all original context and phrasing intact.
- Replace only the value with its equivalent in EUR.

Only return the modified version of the original text.

---

Original Text:
{input_text}
"""
    return PromptTemplate(input_variables=["input_text"], template=prompt_str)


def run_currency_conversion(state: dict) -> dict:
    input_text = state["input_text"]  # consume input_text
    prompt = build_currency_conversion_prompt()
    print("settings.GROQ_API_KEY", settings.GROQ_API_KEY)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=settings.GROQ_API_KEY
    )

    chain: Runnable = prompt | llm
    converted_text = chain.invoke({"input_text": input_text})

    return {
        "converted_text": converted_text  # only add converted_text
    }
