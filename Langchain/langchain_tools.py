from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

@tool
def weather(city:Annotated[str, "被查询的城市"]):
    '''    
    用于查询输入城市今日的天气状况。
    参数:
    city:查询的城市
    '''
    if city == "上海":
        return "上海今日有台风12级"
    else:
        return "天气晴朗，风和日丽"

@tool
def temperature(city:Annotated[str, "被查询的城市"]) -> str:
    '''
    用于查询输入城市今日的温度状况
    参数:
    city:查询的城市
    '''### 工具的docstring必须包括，否则会报错, Annoted是对参数的类型与含义描述
    if city == "上海":
        return "上海今日温度是零下3摄氏度"
    return "温度普遍为12摄氏度"

### 手动将工具嵌入模型    
class tool_llm:
    def __init__(self):
        
        tool_list = [weather, temperature,]
        
        self._llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b"
        ).bind_tools(tool_list)
        
        self._prompt_template = [
            ("system", "你是一个会使用工具的AI助手, 请利用工具来回答问题"),
            ("user", "请帮我查询一下长春的天气和温度如何？")
        ]
        
        
    def __call__(self):
        tool_messages = self._llm.invoke(self._prompt_template)
        self._prompt_template.append(tool_messages)
        if tool_messages is not None:
            for _tool in tool_messages.tool_calls:
                _fn = eval(_tool["name"])
                _tool_res = _fn.invoke(_tool["args"])
                self._prompt_template.append(_tool_res)
        rt = self._llm.invoke(self._prompt_template)
        return rt.content

    
if __name__ == "__main__":
    agent = tool_llm()
    print(agent())
        
#### 自动嵌入模型，直接构建agent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

class Prebuilt_llm:
    def __init__(self):
        llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b"
        )
        
        tool_list = [weather, temperature,]
        
        self._agent_exe = create_react_agent(llm, tool_list)
        
        _prompt = ChatPromptTemplate([
            ("system","你是一个{name}助手，尽可能使用里面所有的工具帮助客户回答问题"),
            ("user","请告诉我今日{city}的温度和天气是什么")
        ])
        # _prompt_value = _prompt.invoke(context)
        
        self._chain = _prompt | self._agent_exe 
        self._parser = StrOutputParser()
        
    def __call__(self):
        context = {"name":"AI", "city":"上海"}
        rt =  self._chain.invoke(context)
        print(rt)
        return self._parser.invoke(rt["messages"][-1])
    
if __name__ == "__main__":
    prebuilt = Prebuilt_llm()
    rt = prebuilt()
    print(rt)
