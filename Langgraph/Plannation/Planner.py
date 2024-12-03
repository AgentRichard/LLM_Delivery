from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class PlannerOutputState(BaseModel):
    query : str = Field(description="用户所提的问题")
    tasks : list[str] = Field(description="你规划后所分解的子问题，这些子问题前后具有强连接性，即每个问题要想回答，必须要知道前一个问题的答案，按照前面所说的规则来划分问题即可，你不用回答")
    
    
_planner_system_prompt="""
你是一个卓越的问题规划专家，你的任务是将客户的问题按照逻辑解题思路进行分段，
比如 客户的问题是：全球GDP的最高的国家的人口是多少？，你可以将问题划分为两部分：
首先第一个问题是 全球GDP最高的国家是哪个国家？ 第二个是这个国家的人口数量是多少？

类似的问题还有 1 + (2 * 3) = ? ，你首先判断是要先算乘除或者括号里的内容即2*3 = 6，
有了上面这个答案之后再算括号外的内容或者加减运算即 6 + 1 = 7

要求：
1. 所划分的子问题之间是一种思维链的方式；
2. 你仅需划分问题，不用自己去解答问题，因为你要将问题传给后面专门解决问题的专家

你的输出应该满足下面的格式(json)：
{output_format}
"""

_planner_human_prompt="""
客户的问题是:
{query}
"""

class Planner:

    def __init__(self, llm):

        _prompt_template = ChatPromptTemplate([
            ("system", _planner_system_prompt),
            ("human", _planner_human_prompt)
        ])
        
        _parser = JsonOutputParser(pydantic_object=PlannerOutputState)
        _prompt_template = _prompt_template.partial(output_format=_parser.get_format_instructions())
        
        self._chain = _prompt_template | llm | _parser
        
    def __call__(self, _init_state):
        ans = self._chain.invoke(_init_state)
        return ans['tasks']
    
if __name__ == "__main__":
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b",
        temperature=0.7
    )
    planner = Planner(_llm)
    # print(planner({"messages":[HumanMessage({"query":"2024年法国跳水项目冠军的家乡在哪里？"})]}))
    print(planner({"query":"2024年法国跳水项目冠军的家乡在哪里？"}))
        