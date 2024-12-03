from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from utils_collector import websearch
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import MessagesPlaceholder
from typing_extensions import Annotated

@tool
def WebSearch(query:Annotated[str, "需要查询的问题描述"]):
    """利用该工具从网上获取客户所查询的相关信息，帮助客户解决问题"""
    return websearch(query)
    
_planner_system_prompt="""
你是一个卓越的信息搜索者与代码编写者，你的任务是根据客户所提问的内容来去网上搜索相关数据，
如果没有搜索到，再次尝试搜索，如果还是搜不到，则自己虚拟一套数据出来，
最后，你需要跟你搜索到的数据利用python matplotlib来进行代码编写，你不用执行，仅需编写代码即可。
要求：
不需要执行代码，紧需写出代码，
最好保证你的数据是网上搜索到的，且是网上出现频率最高的
你可以用以下工具来完成该任务:
{tool_list}

注意事项：
如果从历史信息中的最后一条message返回的内容中包括 FINAL ANSWER， 那么就直接结束
"""

# _planner_human_prompt="""
# {content}
# """

class Planner:
    def __init__(self, _llm):
        tool_list = [WebSearch,]
        
        _prompt_template = ChatPromptTemplate([
            ("system", _planner_system_prompt),
            MessagesPlaceholder(variable_name="messages"),   ### 有这个，确实不需要 human，因为当下一次信息传来时，这里会记载之前的 List of BaseMessages
            # ("human", _planner_human_prompt)               ### 第一次传来的是human message， 第二次传来的就是 [sys, human, AI, tool, AI] 找-1
        ])
        
        _prompt_template = _prompt_template.partial(tool_list=",".join([tools.name for tools in tool_list]))  ## partial的作用就是初始化时就将template中的某些参数的值应用到模板中
        _llm = create_react_agent(_llm, tool_list)
        
        self._chain = _prompt_template | _llm 
        
    def __call__(self, state):
        rt =  self._chain.invoke(state)
        # print(rt)
        return rt["messages"]
        
if __name__ == "__main__":
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b",
        temperature=0.7
    )
    planner = Planner(_llm)
    print(planner({"messages":[("human","获取英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，就完成。")]}))


