import asyncio
import time

async def my_func(name, delay):
    print(f"任务{name} 开始执行")
    await asyncio.sleep(delay)
    print(f"任务{name} 执行完成")
    return f"结果{name}"

async def main():
    tasks = [
        my_func("任务1", 2),
        my_func("任务2", 3),
        my_func("任务3", 1),
    ]
    await asyncio.gather(*tasks)


async def main2():
    print("main2 开始执行")
    await asyncio.sleep(1)
    task = asyncio.create_task(my_func("任务4", 2))
    print("main2 执行完成")
    return "main2 结果"

if __name__ == "__main__":