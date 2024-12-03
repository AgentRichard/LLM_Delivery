from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from utils_collector import websearch
from typing_extensions import Annotated
from operator import add
from langgraph.prebuilt import create_react_agent

def WebSearch(question:Annotated[str, "互联网查询标题"]):
    """你可以使用网络搜索这个工具来帮助客户解决他们的问题。
    """
    return websearch(question)

class HandlerOutputState(BaseModel):
    content : Annotated[str,add] = Field("你每次所输出的内容")

_handler_system_prompt="""
您是一个优秀的子任务执行者，您需要根据子任务的名称和参考信息，完成子任务的信息查询。
您可以使用以下工具来协助您更好的完成该任务：
{tool_list}
"""

_handler_human_prompt="""
参考信息:
{content}
客户的子任务列表:
{tasks}
"""

class Handler:
    def __init__(self, llm):
        
        tool_list = [WebSearch, ]
        
        _prompt_template = ChatPromptTemplate([
            ("system", _handler_system_prompt),
            ("human", _handler_human_prompt)
        ])
        
        _llm_with_tools = create_react_agent(llm, tool_list)
        
        self._parser = StrOutputParser()
        _prompt_template = _prompt_template.partial(tool_list=",".join([tool.__name__ for tool in tool_list]))
        
        self._chain = _prompt_template | _llm_with_tools
        
    def __call__(self, _init_state):
        ans = self._chain.invoke(_init_state)
        return self._parser.invoke(ans["messages"][-1])
    
if __name__ == "__main__":
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b",
        temperature=0.7
    )
    handler = Handler(_llm)
    print(handler({'content':[], 
                   'tasks': ['2024年奥运会中国跳水项目的冠军是谁？', '这位冠军选手的家乡是哪里？']}))
        