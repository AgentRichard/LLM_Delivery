from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class GathererOutputState(BaseModel):
    tasks : list[str] = Field(description="客户们所传来的任务列表")
    final_ans : str
    
    
_gatherer_system_prompt="""
你是一个卓越的信息总结专家，请你根据客户提供的任务列表tasks, 以及客户的答案 content进行总结

例如：(以下例子仅作示范让你理解)
Query:首先第一个问题是 全球GDP最高的国家是哪个国家？ 第二个是这个国家的人口数量是多少？
Ans: 是中国，美国的总人口数量是 14亿

类似的问题还有 1 + (2 * 3) = ? ，你首先判断是要先算乘除或者括号里的内容即2*3 = 6，
有了上面这个答案之后再算括号外的内容或者加减运算即 6 + 1 = 7

要求：
你的回答仅需按照列表的顺序来输出答案，你所写的总结的阅读对象直接就是客户，因此不用将客户视为第三视角。
"""

_gatherer_human_prompt="""
客户的任务列表是:
{tasks}
客户给出的答案是:
{content}
"""

class Gatherer:

    def __init__(self, llm):

        _prompt_template = ChatPromptTemplate([
            ("system", _gatherer_system_prompt),
            ("human", _gatherer_human_prompt)
        ])
        
        self._parser = StrOutputParser()
        
        self._chain = _prompt_template | llm 
        
    def __call__(self, _init_state):
        ans = self._chain.invoke(_init_state)
        return self._parser.invoke(ans)
    
if __name__ == "__main__":
    _llm = ChatOpenAI(
        api_key="ollama",
        base_url="http://0.0.0.0:60000/v1",
        model="qwen2.5:7b",
        temperature=0.7
    )
    gatherer = Gatherer(_llm)
    print(gatherer({'tasks': ['2024年法国跳水项目的冠军是谁？', '这位冠军选手的家乡是哪里？'],
                    'content':'根据搜索结果，2024年法国巴黎奥运会跳水项目的冠军是中国选手全红婵，而她的家乡位于中国广东省湛江市麻章区麻章镇迈合村。因此，我们可以得出答案：2024年法国跳水项目冠军的家乡在中国广东省湛江市麻章区麻章镇迈合村。FINAL_ANS'}))
        