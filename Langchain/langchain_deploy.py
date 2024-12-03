from langchain_community.document_loaders import WebBaseLoader
import bs4

### 右键，页面检查
_loader = WebBaseLoader(
    web_path=["https://www.jiqizhixin.com/articles/2020-12-14-3"],
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(class_ = ["article__content", "article__title"])
    )
)
_doc = _loader.load()
print(_doc)


from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uvicorn
import warnings
warnings.filterwarnings("ignore")

#### 创建模型
_model = ChatOpenAI(
    api_key = "ollama",
    base_url = "http://127.0.0.1:60000/v1",
    model = "qwen2.5:3b",
    temperature = 0.7
)

### 创建模板
_template = ChatPromptTemplate([
    ("system","翻译以下内容为{language}。"),
    ("human", "{content}")
])

### 创建链 OutputParser that parses LLMResult into the top likely string.
_strOutputParser = StrOutputParser()
_chain = _template | _model | _strOutputParser

### 创建http接口
_api = FastAPI(
    title = "翻译",
    version = "1.0",
    description = "大模型智能翻译"
)

### 将内容嵌入端口之中
add_routes(
    app = _api,                    ## 路由端口
    runnable = _chain,             ## 添加可执行文件（任何函数）, 也可以是 _chain.invoke()         
    path = "/translate"            ## path: A path to prepend to all routes. 一个需要在所有路由前追加的路径
)

uvicorn.run(_api, host="0.0.0.0", port=60001)


### 客户端程序来访问 fast api，服务端程序
import requests

response = requests.post(
    url = "http://0.0.0.0:60002/translate/invoke",   ### 如果chain那里直接写了invoke则这里就不需要
    json = {"input":{"language" : "中文", "content":"Sends a post request."}}
    # (optional) A JSON serializable Python object to send in the body of the Request.
)

### 这个是在自己的环境中，发送请求给远程地址与端口，远程处理完毕后获得返回的消息

print(response)
print(response.content)
### b'{"output":"\xe5\x8f\x91\xe9\x80\x81\xe4\xb8\x80\xe4\xb8\xaa\xe8\xaf\xb7\xe6\xb1\x82POST\xe3\x80\x82","metadata":{"run_id":"53c7156e-87ee-49db-b978-6b2e4ef85b68","feedback_tokens":[]}}'
rep = response
print(rep)
rep["output"]