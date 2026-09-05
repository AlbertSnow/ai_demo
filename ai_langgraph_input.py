import time
from typing import TypedDict, Any, List
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

class MockLLM:
    def invoke(self, prompt: str):
        return f"AI Generated: Answer for '{prompt}'"
        
class MockDatabase:
    def get_user_info(self, user_id: str):
        return {"id": user_id, "role": "vip" if "vip" in user_id else "standard"}

class CustomerSupportState(TypedDict):
    query: str
    response: str
    log: List[str]

def node_customer_service(state: CustomerSupportState, config: RunnableConfig, runtime: Runtime) -> dict:
    """
    custer node: show how to get import dependence from config
    """

    user_query = state["query"]

    llm = runtime.context['llm']
    db = runtime.context['db']

    configurable = config.get("configurable", {})
    user_id = config.get("user_id", "guest")

    print(f"\n[Node] start resolve, User ID:{user_id}")

    if not llm or not db:
        return{"response": "System Error: Dependencies not injected!", "log": ["Error"]}

    user_info = db.get_user_info(user_id)
    user_tier = user_info.get("role")
    print(f"[Node] get user tier from DB : ${user_tier}")

    writer = runtime.stream_writer
    writer({"status": "thinking", "message": f"Compile get user generate response from LLM"})
    time.sleep(0.5)

    prompt = f"User{user_tier} asks: {user_query}"
    llm_response = llm.invoke(prompt)

    return {
        "response": llm_response,
        "log": [f"Processed by {llm.__class__.__name__}"]
    }


def run_demo():
    builder = StateGraph(CustomerSupportState)
    builder.add_node("customer_service", node_customer_service)
    builder.add_edge(START, "customer_service")
    builder.add_edge("customer_service", END)
    graph = builder.compile()

    my_llm = MockLLM()
    my_db = MockDatabase()


    print("="*30)
    print("Scene: dependence import demo")
    print("="*30)

    initial_state = {"query": "how to upgrade vip?"}

    config = {
        "configurable": {
            "user_id": "vip_user_999",
        }
    }

    print("[System] Running graph, inject dependence")

    context = {
        "llm": my_llm,
        "db": my_db
    }

    result = graph.invoke(initial_state, config=config, context=context)
    print(result)


if __name__ == "__main__":
    run_demo()
