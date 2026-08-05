import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

def get_ai_response(prompt):
    # response = llm.invoke(prompt)
    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
    )
    print(response.content)
    return response.content


if __name__ == "__main__":
    get_ai_response("What is the capital of China?")