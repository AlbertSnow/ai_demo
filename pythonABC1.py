import PythonABC
import asyncio
import time

print(PythonABC.TestClass.name)
print(PythonABC.TestClass.showClassMethod())
# print(PythonABC.TestClass.showStaticMethod())

print("Execute in pythonABC1.py, name: " + __name__)


async def my_func(name, delay):
    print(f"任务 {name} 开始... ");
    await asyncio.sleep(delay)
    print(f"任务 {name} 结束... ");

async def main_concurrent():
    print(f"并行执行")
    task1 = asyncio.create_task(my_func("c", 1));
    task2 = asyncio.create_task(my_func("B", 4));
    await asyncio.gather(task1, task2);
    print(f"并行执行结束");

if __name__ == "__main__":
    asyncio.run(main_concurrent());