from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from utils_collector import repl
from typing_extensions import Annotated
from langgraph.prebuilt import create_react_agent

@tool
def PythonRepl(input:Annotated[str, "调查者给出的可执行的代码"]):
    """使用这个工具来执行python代码，如果你想看到一个值的输出，你应该用`print(...)`打印出来，这个结果或者绘图结果是用户可见的"""
    try:
        result = repl.run(input)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{input}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.， 如果返回了结果，那么必须加上 FINAL ANSWER")

_drawer_sys_prompt_="""
你是一个卓越的代码执行者，擅长通过工具来执行代码，与检阅并修改代码；
你的任务是先检阅传递过来的代码逻辑，如果执行逻辑不对那么先修改代码再执行，
请你保证能帮助客户完成任务，

你可以使用以下工具:
{tool_list}

如果你收到的内容中有 FINAL ANSWER那么直接结束
"""

class Drawer:
    def __init__(self, _llm):
        tool_list = [PythonRepl, ]
        
        _llm = create_react_agent(_llm, tool_list)
        
        _prompt_template = ChatPromptTemplate([
            ("system", _drawer_sys_prompt_),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        _prompt_template = _prompt_template.partial(tool_list=",".join(tools.name for tools in tool_list))

        self._chain = _prompt_template | _llm
        
    def __call__(self, state):
        rt = self._chain.invoke(state)
        return rt["messages"][-1].content
    
if __name__ == "__main__":
    
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b"
    )
    
    drawer = Drawer(_llm)
    print(drawer({"messages":[(
        "human","根据上述参考信息和推测，假定英国过去5年的GDP数据如下（单位：亿英镑）：\n\n```python\nimport matplotlib.pyplot as plt\n\n# 假设的GDP数据\nyears = [2019, 2020, 2021, 2022, 2023]\ngdp_values = [22000, 21000, 21500, 22000, 22500]\n\n# 绘制GDP时间序列图\nplt.figure(figsize=(10, 6))\nplt.plot(years, gdp_values, marker='o', linestyle='-')\nplt.title('英国过去5年国内生产总值 (单位: 亿英镑)')\nplt.xlabel('年份')\nplt.ylabel('GDP (亿英镑)')\nplt.grid(True)\nplt.show()\n```\n\n这段代码将绘制一个简单的图表，展示假设的英国过去五年的GDP数据。请注意这只是基于推测和参考信息生成的数据示例。为了获得准确数据，建议查阅官方统计数据。"
    )]}))
    