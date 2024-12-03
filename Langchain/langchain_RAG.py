from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from RAG_DOCs import limingxuan, chenxiaolin

model_dir = "/root/project/My_projects/lang_chain/RAG_embedding_model/AI-ModelScope/bge-large-zh-v1___5"
model_name = model_dir
model_kwargs = {"device":"cuda"}
encode_kwargs = {"normalize_embeddings":True}
_emb = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
    query_instruction = "为这个句子生成表示以用于检索相关文章"
)

_docs = [
    Document(page_content=limingxuan),
    Document(page_content=chenxiaolin)
]

text_spliter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_spliter.split_documents(_docs)
_vectorstore = Chroma.from_documents(splits, _emb)       # Chroma 向量存储集成
print(_vectorstore.similarity_search_with_relevance_scores("谁是搞技术的", k=2)) ### 目前不是runnable类型，因此要转换


from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

_llm = ChatOpenAI(
    api_key= "ollama",
    base_url="http://0.0.0.0:60000/v1",
    model="qwen2.5:3b",
    temperature=0.7,
    top_p=0.9
)
_prompt = """
参考以下的内容
{ref}
回答以下问题
{question}
"""
_messages = ChatPromptTemplate([
    ("system", "使用中文帮助用户"),
    ("human", _prompt)
])

# _retriver = _vectorstore.as_retriever().bind(k=2) ## 此时就是runnable的了， 不常用
_retriver = RunnableLambda(func=_vectorstore.similarity_search_with_relevance_scores).bind(k=2) ## 这个常用，转换成runnable类型
## Create a RunnableLambda from a callable, and async callable or both.
## Accepts both sync and async variants to allow providing efficient implementations for sync and async execution.
## 此时RAG就能加到模型中去了

_chain = {"question":RunnablePassthrough(), "ref":_retriver}| _messages | _llm | StrOutputParser()
## RunnablePassthrough 我输的函数在内部进行参考 

rt = _chain.invoke("陈晓琳的家乡在哪里")
print(rt)

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

runnable = RunnableParallel(
    origin=RunnablePassthrough(),
    modified=lambda x: x+1
)
print(runnable)
runnable.invoke(1) # {'origin': 1, 'modified': 2}
d = {"question":RunnablePassthrough(), "ref":_retriver}
print(d)