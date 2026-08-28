from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command

# custom reducer
def add_message(message_list_left:list, message_list_right:list):
    print("="*20)
    print("Running add message")
    print("Left: ", message_list_left)
    print("Right: ", message_list_right)
    print("="*20)
    return message_list_left + message_list_right

class MyAgent(TypedDict):
    # init custom reducer
    messages: Annotated[List[BaseMessage], add_message]

def tool_node(state: MyAgent):
    return {"messages": [ToolMessage(content="It's from tool_node content", tool_call_id="tool_call_id_1")]}

def llm_node(state:MyAgent):
    return {"messages": [AIMessage(content="It's from llm_node content")]}

builder = StateGraph(state_schema=MyAgent)
builder.add_node("tool_node", tool_node)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "tool_node")
builder.add_edge("tool_node", "llm_node")
builder.add_edge("llm_node", END)
compiled_graph = builder.compile()
res = compiled_graph.invoke({"messages":[HumanMessage(content="Hello world")]})
print("最终结果：",res)

## default Reducer (cover update)
# class DefaultReducerState(TypedDict):
#     foo: int
#     bar: List[str]

# def node_default_1(state: DefaultReducerState) -> dict:
#     return {"foo": 2}
# def node_default_2(state: DefaultReducerState) -> dict:
#     return {"bar": ["bye"]}

# def run_demo():
#     print("1. Default Reducer Demo: ")
#     builder = StateGraph(state_schema=DefaultReducerState)
#     builder.add_node("node_default_1", node_default_1)
#     builder.add_node("node_default_2", node_default_2)
#     builder.add_edge(START, "node_default_1")
#     builder.add_edge("node_default_1", "node_default_2")
#     builder.add_edge("node_default_2", END)
#     compiled_graph = builder.compile()
#     res = compiled_graph.invoke({"foo": 1, "bar": ["albert"]})
#     print(f"初始状态：foo={1}, bar={['albert']}")
#     print("最终结果：",res)

# if __name__ == "__main__":
#     run_demo()


# class MyStateFull(TypedDict):
#     rag_result: str
#     web_search_result: str
#     final_answer: str
#     query: str
#     a_new_key: str

# class MyState(TypedDict):
#     query: str
#     final_answer: str

# class SearchState(TypedDict):
#     rag_result: str
#     web_search_result: str

# class InputSchema(TypedDict):
#     query: str

# class OutputSchema(TypedDict):
#     final_answer: str

# graph = StateGraph(state_schema=MyStateFull, input_schema=InputSchema, output_schema=OutputSchema)

# def rag_search_node(state:MyState):
#     print("rag_search_node 执行", state)
#     query = state["query"]
#     rag_result = f"关于{query}的rag_result"
#     return {"rag_result": rag_result, "a_new_key": "a_new_key_value"}

# def web_search_node(state:MyState):
#     print("web_search_node 执行", state)
#     query = state["query"]
#     web_search_result = f"关于{query}的web_search_result"
#     return {"web_search_result": web_search_result}

# def final_answer_node(state:MyStateFull):
#     rag_result = state["rag_result"]
#     web_search_result = state["web_search_result"]
#     final_answer = f"LLM基于{rag_result}和{web_search_result}的最终答案"
#     return {"final_answer": final_answer}

# graph.add_node(rag_search_node)
# graph.add_node(web_search_node)
# graph.add_node(final_answer_node)

# graph.add_edge(START, "rag_search_node")
# graph.add_edge(START, "web_search_node")
# graph.add_edge("rag_search_node", "final_answer_node")
# graph.add_edge("web_search_node", "final_answer_node")
# graph.add_edge("final_answer_node", END)

# compiled_graph = graph.compile()
# res = compiled_graph.invoke({"query": "什么是人工智能"})
# print("最终结果：",res)

# app = StateGraphExecutor(graph)

# result = app.invoke({"query": "什么是人工智能"})
# print(result)