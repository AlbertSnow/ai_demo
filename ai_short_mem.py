
from langchain_core.messages import HumanMessage
from langchain_protocol.protocol import ToolsData


def scenario_1_demo():
    import os
    from dotenv import load_dotenv
    from pydantic import SecretStr
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    from langgraph.checkpoint.memory import InMemorySaver


    load_dotenv()

    llm = ChatOpenAI(
        model=os.getenv("MODEL") or "",
        api_key=SecretStr(os.getenv("API_KEY") or ""),
        base_url=os.getenv("BASE_URL") or "",
        temperature=1.0
    )

    checkpoint = InMemorySaver()

    @tool
    def weather_tool(city: str, date: str) -> str:
        """查询指定城市在指定日期的天气。"""
        return f"The weather in {city} on {date} is sunny."

    agent = create_agent(llm, [weather_tool], checkpointer=checkpoint)

    user1_res = agent.invoke(
    {"messages": [HumanMessage(content="北京2026-08-22的天气怎么样")]},
    config={"configurable": {"thread_id": "user1_session1"}},
    )
    print('第一次调用', user1_res['messages'][-1], end="\n\n==============\n\n")    

    user1_res2 = agent.invoke({"messages": [HumanMessage(content="适合出去玩吗？")]},
    config={"configurable": {"thread_id": "user1_session1"}})
    print('第二次调用', user1_res2['messages'][-1], end="\n\n==============\n\n")

if __name__ == "__main__":
    scenario_1_demo()