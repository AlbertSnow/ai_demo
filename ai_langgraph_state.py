from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MyStateFull(TypedDict):
    rag_result: str
    web_search_result: str
    final_answer: str
    query: str
    a_new_key: str

class InputSchema(TypedDict):
    query: str

class OutputSchema(TypedDict):
    final_answer: str

graph = StateGraph(state_schema=MyStateFull, input_schema=InputSchema, output_schema=OutputSchema)

def rag_search_node(state:MyStateFull):
    print(state)
    query = state["query"]
    rag_result = f"关于{query}的rag_result"
    return {"rag_result": rag_result, "a_new_key": "a_new_key_value"}


def web_search_node(state:MyStateFull):
    print(state)
    query = state["query"]
    web_search_result = f"关于{query}的web_search_result"
    return {"web_search_result": web_search_result}

def final_answer_node(state:MyStateFull):
    print('在final_answer_node中，state: ', state)
    rag_result = state["rag_result"]
    web_search_result = state["web_search_result"]
    final_answer = f"LLM基于{rag_result}和{web_search_result}的最终答案"
    return {"final_answer": final_answer}

graph.add_node(rag_search_node)
graph.add_node(web_search_node)
graph.add_node(final_answer_node)

graph.add_edge(START, "rag_search_node")
graph.add_edge(START, "web_search_node")
graph.add_edge("rag_search_node", "final_answer_node")
graph.add_edge("web_search_node", "final_answer_node")
graph.add_edge("final_answer_node", END)

compiled_graph = graph.compile()
res = compiled_graph.invoke({"query": "如何使用LangGraph", "a_new_key": "a_new_key_value"})
print('final_result: ', res)