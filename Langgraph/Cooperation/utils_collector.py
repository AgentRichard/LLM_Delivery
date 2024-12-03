## 1. WebSearch
from zhipuai import ZhipuAI

class WebSearch:
    def __init__(self):
        self._client = ZhipuAI(api_key="50d0b340e21e2af2491369524cc6ad1c.FvQYe9TMdvXIaStL")
        self._tools = [{
            "type": "web_search",
            "web_search": {
                "enable": True,
            }
        }]
        
    def __call__(self, query):
        messages = [{
            "role": "user",
            "content": query
        }]
        response = self._client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
            tools=self._tools
        )
        return response.choices[0].message.content
    
websearch = WebSearch()

## 2. WebContentCollector
from langchain_community.document_loaders import WebBaseLoader
import bs4 #beautifulsoup4
from typing_extensions import Annotated

# class WebDocumentLoader:
#     def __init__(self,  web_url: Annotated[list[str], "List of web URLs"], content_class: Annotated[list[str], "List of CSS classes to parse"]):
#         self._loader = WebBaseLoader(
#             web_path=web_url,
#             bs_kwargs=dict(parse_only = bs4.SoupStrainer(class_ = content_class))
#             )
        
#     def __call__(self):
#         content = []
#         for _docs in self._loader.load():
#             content.append(_docs.page_content)
#         return content
    
# web_document_loader = WebDocumentLoader(["https://www.jiqizhixin.com/articles/2020-12-14-3"],
#                                         ["article__content", "article__title"])

## 3. RAG
# from RAG_DOCs import limingxuan, chenxiaolin
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# class RAG_vectorstore_score:
#     def __init__(self, model_dir:str):
#         _model_name = model_dir
#         _model_kwargs = {"device":"cuda"}
#         _encode_kwargs = {"normalize_embeddings":True}
        
#         _emb = HuggingFaceBgeEmbeddings(
#             model_name = _model_name,
#             model_kwargs = _model_kwargs,
#             encode_kwargs = _encode_kwargs,
#             query_instruction = "为这个句子生成表示以用于检索相关文章" 
#         )
        
#         _docs = [
#             Document(page_content=limingxuan),
#             Document(page_content=chenxiaolin)
#         ]
        
#         _text_spliter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#         _splits = _text_spliter.split_documents(_docs)
#         self._vectorstore = Chroma.from_documents(_splits, _emb)
        
#     def __call__(self, _query:str, _k:int):
#         _retriver = RunnableLambda(func=self._vectorstore.similarity_search_with_relevance_scores).bind(k=_k)
#         return _retriver
    
# rag_similarity_search_scores = RAG_vectorstore_score(model_dir="/root/project/My_projects/lang_graph/RAG_embedding_model/AI-ModelScope/bge-large-zh-v1___5")

## 4. code executor
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()

## 5. weather
class Weather:
    def __init__(self):
        pass
    def __call__(self, city:Annotated[str, "要被查询的城市名称"]):
        if city == "上海":
            return "今日上海的天气阳光明媚"
        return f"今日{city}的天气乌云密布"
    
weather = Weather()

## 6. 
# from RAG_DOCs import limingxuan, chenxiaolin
# from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_community.embeddings import OllamaEmbeddings

# class RAG:
#     def __init__(self):
        
#         _emb = OllamaEmbeddings(
#             base_url = "http://0.0.0.0:60000",
#             model = "bge-m3"
#         )
        
#         _docs = [
#             Document(page_content=limingxuan),
#             Document(page_content=chenxiaolin)
#         ]

#         _splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
#         _splits = _splitter.split_documents(_docs)
#         self._vectorstore = Chroma.from_documents(_splits, _emb)
        
#     def __call__(self, query, k):
#         return self._vectorstore.similarity_search_with_relevance_scores(query=query, k=2)

# rag_retriever = RAG()