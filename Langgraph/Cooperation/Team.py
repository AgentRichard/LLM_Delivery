from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from Planner import Planner
from Drawer import Drawer
from langchain_core.messages import HumanMessage

class Team:
    def __init__(self):
        _llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b",
            temperature=0.7
        )
        
        self._planner = Planner(_llm)
        self._drawer = Drawer(_llm)
        self._graph = self._init_graph()
    
    def _init_graph(self):
        _builder = StateGraph(MessagesState)
        _builder.add_node("planner", self._agent_planner)
        _builder.add_node("drawer", self._agent_drawer)
        
        _builder.add_edge(START, "planner")
        _builder.add_conditional_edges("planner", self._router, {"continue":"drawer", END:END})
        _builder.add_conditional_edges("drawer", self._router, {"continue":"planner", END:END})
        
        _graph = _builder.compile()
        return _graph
    
    def _agent_planner(self, state):
        tmp_res = self._planner(state)
        return {"messages":[tmp_res[-1]]}
    
    def _agent_drawer(self, state):
        tmp_res = self._drawer(state)
        return {"messages":[tmp_res[-1]]}
    
    def _router(self, state):
        _last_messages = state["messages"][-1]
        if "FINAL ANSWER" in _last_messages.content.upper():
            return END
        else:
            return "continue"
    
    def __call__(self, _init_state):
        result =  self._graph.invoke(_init_state)
        return result["messages"][-1].content
    
if __name__ == "__main__":
    team = Team()
    print(team({"messages":[HumanMessage("获取英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，就完成。")]}))