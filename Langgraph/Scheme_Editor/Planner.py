from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class SubState(BaseModel):
    title : str = Field(description="一级目录")
    subtitles : list[str] = Field(description="二级目录")

class State(BaseModel):
    project_name : str = Field(description="项目的名称")
    abstract : str = Field(description="项目的摘要")
    outlines : list[SubState] = Field(description="项目的一级目录与二级目录列表")
    
_planner_system_prompt="""
你是一个卓越的方案撰写师，你的任务是根据客户提出的方案名称来编写方案摘要；
并根据方案名称与摘要设计多个一级目录的标题，并且设计一级目录下多个二级子目录的标题，
不用写这些目录内的内容，因为你进负责摘要，标题这一部分，
编写的要求：
1. 不要为这些标题加序号，
2. 不要过度超出标题与摘要的范围，
3. 每个一级目录最少要求三个二级子目录
输出格式：
{output_format}
"""

_planner_human_prompt="""
客户的方案名称：
{project_name}
"""

class Planner:
    def __init__(self, _llm):
        
        _prompt_template = ChatPromptTemplate([
            ("system", _planner_system_prompt),
            ("human", _planner_human_prompt)
        ])
        
        _parser = JsonOutputParser(pydantic_object=State)
        _prompt_template = _prompt_template.partial(output_format=_parser.get_format_instructions())
        
        self._chain = _prompt_template | _llm | _parser
        
    def __call__(self, state):
        return self._chain.invoke(state)
    
if __name__ == "__main__":
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b",
        temperature=0.7
    )
    planner = Planner(_llm)
    print(planner({"project_name":"基于多模态驱动的数字人"}))