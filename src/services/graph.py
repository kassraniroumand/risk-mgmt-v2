"""
the module defines an async factory function that builds a langchain dag for
orchestrating a sequence and parallel nodes related analysis pipelines

Key point
----
1. GraphState: a typedDict that define the schema shared across all nodes
2. nodes: individual service function
3. edges: define the workflow between the nodes
4. end: indicate the end of workflow

IMPORTANT:
first thing I did here is to unify the currency in the report then
further process start on the unified document

"""
from src.services.insurance_recommendation import run_insurance_recommendation
from src.services.multi_currency_risk import run_multy_currency_risk
from src.services.currency_convertion import run_currency_conversion
from src.services.current_insurance import run_current_insurance
from src.services.business_interruption import run_business_interruption
from src.services.property_valudation import run_property_valuation
from src.services.risk_percentages import run_risk_percentage
from typing import TypedDict, Annotated, List, Any
from langgraph.graph import StateGraph, END



class GraphState(TypedDict):
    """Central state object passed between graph nodes"""
    input_text: str
    converted_text: str
    property_valuations_s: str
    risk_percentage_s: str
    business_interruption_s: str
    current_insurance_s: str
    multi_currency_risk_s: str
    insurance_recommendation_s: str

async def create_graph() -> Any:
    """compile and return a Langchian DAG"""
    graph = StateGraph(GraphState)

    # define the add and its related id that should match the id used in edge
    graph.add_node("convert_currency", run_currency_conversion)
    graph.add_node("property_valuation", run_property_valuation)
    graph.add_node("risk_percentage", run_risk_percentage)
    graph.add_node("business_interruption", run_business_interruption)
    graph.add_node("current_insurance", run_current_insurance)
    graph.add_node("multi_currency_risk", run_multy_currency_risk)
    graph.add_node("insurance_recommendation", run_insurance_recommendation)

    # entry point to graph
    graph.set_entry_point("convert_currency")

    # define the relation between nodes and make the workflow parallel to reduce latency
    graph.add_edge("convert_currency", "property_valuation")
    graph.add_edge("convert_currency", "risk_percentage")
    graph.add_edge("convert_currency", "business_interruption")
    graph.add_edge("convert_currency", "current_insurance")
    graph.add_edge("convert_currency", "multi_currency_risk")
    graph.add_edge("convert_currency", "insurance_recommendation")

    graph.add_edge("property_valuation", END)
    graph.add_edge("risk_percentage", END)
    graph.add_edge("business_interruption", END)
    graph.add_edge("current_insurance", END)
    graph.add_edge("multi_currency_risk", END)
    graph.add_edge("insurance_recommendation", END)

    # Compile and return
    dag = graph.compile()
    return dag