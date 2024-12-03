### 规划要解决的问题就是模型无法思考
### 对于一个全新的知识，假设大模型是2023年出的，2024年的信息他并不知道，但是网上的内容有
### 模型可以通过网上的内容以思维链的方式去做合理推断
### 对于新知识或者数学知识的处理，只有两种办法，第一种是微调，第二种是思维链
### 方案规划问题是信息独立的，因此可以并行处理，但是思维链必须通过第一个问题的答案来处理后面的答案
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import TypedDict, Annotated
from Planner import Planner
from Handler import Handler
from Gatherer import Gatherer
from operator import add

class State(TypedDict):
    query : str
    tasks : list[str]
    content : Annotated[list[str], add]
    final_ans : str
    
class Team:
    def __init__(self):
        _llm = ChatOpenAI(
            api_key="ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:7b",
            temperature=0.7
        )
        
        self._planner = Planner(_llm)
        self._handler = Handler(_llm)
        self._gatherer = Gatherer(_llm)
        self._graph = self._init_graph()
        
    def _init_graph(self):
        _builder = StateGraph(State)
        _builder.add_node("planner", self._agent_planner)
        _builder.add_node("handler", self._agent_handler)
        _builder.add_node("gatherer", self._agent_gatherer)
        
        _builder.add_edge(START, "planner")
        _builder.add_edge("planner", "handler")
        _builder.add_conditional_edges("handler", self._backward_edge)
        _builder.add_edge("gatherer", END)
        
        _graph = _builder.compile()
        return _graph
    
    def _agent_planner(self, state):
        tmp = self._planner(state)
        return {"tasks":tmp}
        
    
    def _agent_handler(self, state):
        while True:
            try:
                _content = state.get("contents", [])
                _tasks_index = len(_content)
                _task = state["tasks"][_tasks_index]
                tmp = self._handler({
                    "content":_content,
                    "tasks":_task
                })
                return {"content":[tmp]}
            except Exception as e:
                print(e)

    def _agent_gatherer(self, state):
        while True:
            try:
                tmp = self._gatherer(state)
                return {"final_ans":tmp}
            except Exception as e:
                print(e)
    
    def _backward_edge(self, state):
        _task_len = len(state["tasks"])
        _content = len(state["content"])
        if _task_len == _content:
            return "handler"
        return "gatherer"        
    
    def __call__(self, _init_state):
        result = self._graph.invoke(_init_state)
        return result
    
if __name__ == "__main__":
    team = Team()
    # result = team({"query":"2024年法国奥运会女子10米跳水项目金牌夺得者的家乡在哪里?并且告诉我年龄和过往培训的记录,除此之外我还想知道她父母是干什么的，她的队友是谁?"})
    result = team({"query":"2024年法国奥运会女子10米跳水项目金牌夺得者的家乡在哪里?"})
    print(result["query"])
    print(result["tasks"])
    print(result["content"])
    print(result["final_ans"])