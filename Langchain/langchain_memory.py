from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class memory_:
    def __init__(self):
        _llm = ChatOpenAI(
            api_key= "ollama",
            base_url="http://0.0.0.0:60000/v1",
            model="qwen2.5:3b",
            temperature=0.7,
            top_p=0.9
        )
    
        _prompt_template = ChatPromptTemplate([
            ("system", "你作为一个{name1}助手，帮助客户回答问题"),
            MessagesPlaceholder(variable_name="history_content"),
            ("user", "{content}")
        ])
        print(_prompt_template)
        self.store = {}
        _parser = StrOutputParser()
        
        self._chain = _prompt_template | _llm | _parser ## 不用加parser，因为我们要先取到最后一个回复
        
        self.history_chain = RunnableWithMessageHistory(
            runnable=self._chain,
            get_session_history=self.get_session_history,
            input_messages_key="content",
            history_messages_key="history_content"
        )
        print(self.history_chain)
    def get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def __call__(self):
        config = {"name1":"AI", "content":"记住我的名字叫张三"}
        res = self.history_chain.invoke(input=config, config={"configurable":{"session_id":"1"}})
        print(res)
        
        config = {"name1":"AI", "content":"我叫什么名字"}
        res = self.history_chain.invoke(input=config, config={"configurable":{"session_id":"1"}})
        # print(self.store)
        print(res)
    
if __name__ == "__main__":
    me = memory_()
    re = me()