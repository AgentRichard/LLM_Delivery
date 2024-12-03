from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class WritterSubScheme(BaseModel):
    content : str = Field(description="你所生成的内容")
    
_writter_system_prompt="""
你是一个卓越的方案编写师，你的任务是根据客户给出的方案名称、方案摘要、方案的一级目录与方案二级目录进行方案内容撰写；
你所写的内容应该完整符合项目的各个标题，不要越界，除此之外，你所写的内容应该具备专业度；
要求：
不要加序号；
要尽可能完整，且高度逻辑
输出要求：
{output_format}
"""

_writter_human_prompt="""
客户的方案名称：
{project_name}
客户的方案摘要:
{abstract}
客户方案的一级目录:
{title}
客户方案的二级目录：
{subtitles}
"""

class Writter:
    def __init__(self, _llm):
        
        _prompt_template = ChatPromptTemplate([
            ("system", _writter_system_prompt),
            ("human", _writter_human_prompt)
        ])
        
        _parser = JsonOutputParser(pydantic_object=WritterSubScheme)
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
    writter = Writter(_llm)
    rt = writter({'project_name': '基于多模态驱动的数字人', 
                   'abstract': '该项目旨在通过整合语音、文本和视觉等多种输入模态，开发一个能够实现自然交互的虚拟人物。该数字人将在客户服务、教育培训等领域提供更加生动、智能的服务体验。', 
                   'title': '项目背景与意义', 
                   'subtitles': ['多模态技术的发展现状', '市场需求分析', '预期目标和挑战']})
    print(rt["content"])