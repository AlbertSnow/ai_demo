import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

llm = Tongyi()
chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr(os.getenv("DEEPSEEK_API_KEY") or ""))
text = "给生产杯子的公司取一个名字，直接输出最终的名字"
messages = [HumanMessage(content=text)]

if __name__ == "__main__":
    print(llm.invoke(text))
    print("--------------------------------")
    print(chat_model.invoke(messages))