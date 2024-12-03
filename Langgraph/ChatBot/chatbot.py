from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph.message import MessagesState
from langchain_core.tools import tool
from utils_collector import websearch, rag_retriever, weather
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from uuid import uuid4

@tool
def Weather(city:Annotated[str, "需要被查询天气的城市名称"]):
    """
    如果涉及天气相关内容，使用此工具进行输出来帮助客户完成任务
    """
    return weather(city=city)

@tool
def WebSearch(query:Annotated[str, "需要上网查询的网页信息"]):
    """可以使用chatbot来搜索网页信息， 并打印出其中一个你搜索的网页的地址"""
    return websearch(query=query)

@tool
def RAG_Vectors(query:Annotated[str, "需要查询知识向量库的有关人物的信息"], _k:Annotated[int, "所查询内容与知识库向量距离最近的最相关的前k个"]):
    '''如果用户查询的信息是关于人物的信息，使用chatbot_retrieve工具'''
    return rag_retriever(query, _k)

_chatbot_system_template="""
你是一个情商很高且智商很高的对话者，请你根据你的知识储备精心帮助客户回答问题，
如果客户所提的问题与真实事件相关，那么请你调用相关工具来回答客户的问题，
你可用以下工具来帮助客户:
{tool_list}
"""

_chatbot_human_template="""
用户的问题是:
{content}
"""


class ChatBot:
    def __init__(self):
        
        self._tool_list = [Weather, WebSearch, RAG_Vectors, ]
        
        self._llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b",
            temperature=0.7
        ).bind_tools(self._tool_list)
        
        self._graph = self._init_graph()
        
    def _init_graph(self):
        _builder = StateGraph(MessagesState)   ## messages: Annotated[list[AnyMessage], add_messages] 状态已经被初始化好
        _builder.add_node("chat_agent", self._chat_agent)
        _builder.add_node("tool_agent", ToolNode(tools=self._tool_list))
        
        _builder.add_edge(START, "chat_agent")
        _builder.add_conditional_edges("chat_agent", self._conditional_tool_edge, )
        _builder.add_edge("tool_agent", "chat_agent")
        
        _memory = MemorySaver()
        _graph = _builder.compile(_memory)
        return _graph
    
    def _conditional_tool_edge(self, state):
        _last_message = state["messages"][-1]
        if _last_message.tool_calls:
            return "tool_agent"
        return END
        
    def _chat_agent(self, state):
        res = self._llm.invoke(state["messages"])
        return {"messages":[res]}  ## 因为MessagesState是add_messages， 因此即使返回的仅有一个Ai messages，为了规范化也要加中括号
    
    
    def _graph_draw(self):
        display(Image(self._graph.get_graph(xray=10).draw_mermaid_png()))
    
    def __call__(self):
        _thread_id = uuid4()
        while True:
            _human = input("我: ")
            if _human == "/bye":
                print("AI: Bye")
                return
            _config = {"thread_id":_thread_id, "recursion_limit":20, "configurable":{"session_id":"1"}}
            rt = self._graph.invoke({"messages":[HumanMessage(_human)]}, config=_config)
            print("AI-Assistant: ", rt["messages"][-1].content)
    
if __name__ == "__main__":
    chatbot = ChatBot()
    rt = chatbot()
    chatbot._graph_draw()