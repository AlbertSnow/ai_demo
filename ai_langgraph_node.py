from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage,HumanMessage,SystemMessage,AIMessage,BaseMessage
from typing import Annotated, TypedDict, List

def add_message(message_list_lift:list, message_list_right:list):
    print("="*20)
    print("正在执行 add_message")
    print('左边的', message_list_lift)
    print('右边的', message_list_right)
    print("="*20)
    return message_list_lift + message_list_right

class MyAgent(TypedDict):
    messages:Annotated[List[BaseMessage], add_message]

def tool_node(state:MyAgent):
    return {"messages": [ToolMessage(content="tool_node执行结果", tool_call_id="tool_call_id")]}


def llm_node(state:MyAgent):
    return {"messages": [AIMessage(content="llm_node执行结果")]}

builder = StateGraph(state_schema=MyAgent)
builder.add_node("tool_node", tool_node)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "tool_node")
builder.add_edge("tool_node", "llm_node")
builder.add_edge("llm_node", END)

compiled_graph = builder.compile()
res = compiled_graph.invoke({"messages": [HumanMessage(content="你好")]})
print('final_result: ', res)