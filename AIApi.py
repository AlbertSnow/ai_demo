import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

def get_ai_response(prompt):
    response = llm.invoke(prompt)
    print(response.content)
    return response.content


if __name__ == "__main__":
    get_ai_response("What is the capital of France?")