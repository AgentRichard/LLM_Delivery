from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from Planner import Planner
from Writter import Writter
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Any, TypedDict
from operator import add
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

class SubSchemeState(TypedDict):
    title : str
    subtitles : list[str]

class SchemeState(TypedDict):
    project_name : str 
    abstract : str
    outlines : list[SubSchemeState] 
    content : Annotated[list[str], add]
    article : str

class TeamDesigner:
    def __init__(self):
        
        _llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b",
            temperature=0.7
        )
        
        self._planner = Planner(_llm)
        self._writter = Writter(_llm)
        self._graph = self._init_graph()
        
    def _init_graph(self):
        _builder = StateGraph(SchemeState)
        _builder.add_node("planner", self._agent_planner)
        _builder.add_node("writter", self._agent_writter)
        _builder.add_node("gatherer", self._agent_gatherer)
        
        _builder.add_edge(START, "planner")
        _builder.add_conditional_edges("planner", self._dispatch_task)
        _builder.add_edge("writter", "gatherer")
        _builder.add_edge("gatherer", END)
        
        _graph = _builder.compile() ## 不需要memory
        return _graph
    
    def _agent_planner(self, state:SchemeState):
        while True:
            try:
                tmp_res = self._planner(state)
                return tmp_res
            except Exception as e:
                print(e)
    
    def _agent_writter(self, state:SchemeState):
        while True:
            try:
                return {"content":[self._writter(state)["content"]]}
            except Exception as e:
                print(e)
    
    def _agent_gatherer(self, state:SchemeState):
        while True:
            try:
                state["article"] = f"{state["project_name"]}\n\n\n\n"
                state["article"] += f"{state["abstract"]}\n\n\n\n"
                count = 0
                for _index, _outline in enumerate(state["outlines"]):
                    state["article"] += f"{_index+1}. {_outline["title"]}\n\n"
                    for _j, _subtitles in enumerate(_outline["subtitles"]):
                        state["article"] += f"{_index+1}.{_j+1}. {_subtitles}\n\n"
                        state["article"] += f"{state["content"][count]}\n\n\n"
                        count += 1
                return {"article" : state["article"]}
            except Exception as e:
                print(e)
    
    def _dispatch_task(self, state):
        _project_name = state["project_name"]
        _abstract = state["abstract"]
        _pkg = []
        for _outline in state["outlines"]:
            _title = _outline["title"]
            for _subtitle in _outline["subtitles"]:
                _pkg.append(Send("writter", {
                    "project_name":_project_name,
                    "abstract" : _abstract,
                    "title" : _title,
                    "subtitles" : _subtitle
                }))
        # print(_pkg)
        return _pkg
    
    def __call__(self, _init_state):
        return self._graph.invoke(_init_state)
    
if __name__ == "__main__":
    team = TeamDesigner()
    rt = team({"project_name":"中国和台湾会打仗吗？"})
    with open("artical.txt", "w+") as fw:
        fw.write(rt["article"])