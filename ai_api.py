import os
import io
import asyncio
import base64
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import SecretStr
from PIL import Image
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL") or "",
    api_key=SecretStr(os.getenv("API_KEY") or ""),
    base_url=os.getenv("BASE_URL") or "",
    temperature=1.0
)

vision_llm = ChatOpenAI(
    model=os.getenv("QWEN_VISION_MODEL") or "qwen-vl-plus",
    api_key=SecretStr(os.getenv("QWEN_API_KEY") or ""),
    base_url=os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
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

def get_ai_conversation():
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. You name is Json"),
        HumanMessage(content="My name is Felx"),
        AIMessage(content="Hello. How can I help you today?"),
        HumanMessage(content="What is the name of mine.")
    ])
    print(response.content)

def get_tuple_conversation():
    print("profile: ", llm.profile)
    with get_usage_metadata_callback() as cb:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant. You name is Json"},
            {"role": "user", "content": "My name is Felx"},
            {"role": "assistant", "content": "Hello. How can I help you today?"},
            {"role": "user", "content": "What is the name of mine."}
        ])
        print(response.content)
        print("print metadata:")
        print(cb.usage_metadata)


def get_dict_conversation():
    response = llm.invoke([
            ("system", "You are a helpful assistant. You name is Json"),
            ("user", "My name is Felx"),
            ("assistant", "Hello. How can I help you today?"),
            ("user", "What is the name of mine.")
        ]
    )
    print(response.content)

def get_template_conversation():
    prompt_template = [
        {"role": "system", "content": "You are a helpful assistant. You name is {AiName}"},
        {"role": "user", "content": "My name is {userName}"},
    ]
    messages = [
        {
            "role": t["role"],
            "content": t["content"].format(AiName="Json", userName="Felx")
        }
        for t in prompt_template
    ]
    response = llm.invoke(messages)
    print(response.content)

async def get_async_conversation():
    response = await llm.ainvoke([
        {"role": "system", "content": "You are a helpful assistant. You name is Json"},
        {"role": "user", "content": "My name is Felix"},
    ])
    print(response.content)

def get_stream_response():
    response = llm.stream("写一首关于爱情的诗")

    print("---------------Stream-----------------")
    print("开始生成...")

    full_message = None
    for chunk in response:
        full_message = chunk if full_message is None else full_message + chunk
        print(chunk.content, end="", flush=True)
    print("完整消息: ", full_message)

async def get_async_stream_event_responce():
    full_token = None
    async for event in llm.astream_events("你好！"):
        if event["event"] == "on_chat_model_start":
            print(f"输入：{event['data']['input']}")
        elif event["event"] == "on_chat_model_stream":
            this_content = event['data']['chunk'].content
            full_token = this_content if full_token is None else full_token + this_content
            print(f"Token: {this_content}", end="", flush=True)
        elif event["event"] == "on_chat_model_end":
            print(f"\nOver: {full_token}")

def get_batch_response():
    questions = [
        "What is the capital of China?",
        "What is the capital of Japan?",
        "What is the capital of Korea?",
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]
    response = llm.batch(questions)
    for question, response in zip(questions, response):
        print(f"Question: {question}")
        print(f"Response: {response.content}")



def _encode_image(image_path: str, max_size: int = 1024) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_picture_response(image_path: str = "view_pic.jpeg", prompt: str = "描述这张图片"):
    encoded_string = _encode_image(image_path)
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}},
        ])
    ]
    response = vision_llm.invoke(messages)
    print(response.content)
    return response.content


if __name__ == "__main__":
    # get_ai_response("What is the capital of China?")
    # get_ai_conversation()
    print("---------------Tuple-----------------")
    get_tuple_conversation()
    # print("---------------Dict-----------------")
    # get_dict_conversation()
    # print("---------------Template-----------------")
    # get_template_conversation()
    # print("---------------Async-----------------")
    # asyncio.run(get_async_conversation())
    # print("---------------Stream-----------------")
    # get_stream_response()
    # print("---------------Async Stream Event-----------------")
    # asyncio.run(get_async_stream_event_responce())
    # print("---------------Batch-----------------")
    # get_batch_response()
    # print("---------------Picture-----------------")
    # get_picture_response()
    print("---------------End-----------------")