#### 本项目可以使用qwen2.5:3b模型来回答以下问题
#### 徐怡雯的家乡在哪里，上的什么学校，家乡的天气如何？重新提问：刚刚主人公的家乡在哪里？

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Annotated
from langchain_core.messages import (HumanMessage, 
                                     SystemMessage,
                                     ToolMessage)
from langchain_core.prompts import (ChatPromptTemplate, 
                                    HumanMessagePromptTemplate, 
                                    SystemMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableLambda, 
                                      RunnablePassthrough, 
                                      RunnableWithMessageHistory)
from langchain_community.chat_message_histories import ChatMessageHistory

# 调用模型
model = ChatOpenAI(
    api_key = "ollama",
    model = "qwen2.5:3b",
    base_url = "http://0.0.0.0:60000/v1",
    temperature = 0.7,
    top_p = 0.8
)

#### 首先实现调用工具的调用
# 创建工具
@tool
def weather(city:Annotated[str, "参数是某一个城市"]):
    """
    这是一个工具，用来查询某个城市的天气状况！
    参数:
    city:某一个城市
    """# doc_str 必写
    if city == "上海":
        return "天气阳光明媚，万里无云"
    return "倾盆大雨，乌云密布"

@tool
def ecomomy(city:Annotated[str,"参数是某一个城市"]):
    """
    这是一个工具，用来查询某个城市的经济状况！
    参数:
    city:某一个城市
    """
    if city == "上海":
        return "这所城市的经济良好，通胀率适中"
    return "经济萎靡不振"


#### 创建RAG库
# 现成文本
person = """
徐怡雯， 女， 出生日期：1998.12.16, 家乡:上海浦东川沙镇，
少年时期
小学：就读于川沙镇中心小学，成绩优秀，尤其擅长语文。她经常参加学校的作文比赛，并多次获奖。在父母的鼓励下，她开始尝试写短篇小说。
初中：升入川沙镇第二中学后，陈晓琳不仅成绩优异，还积极参与学校的文学社活动。她担任文学社的主编，负责校刊的编辑工作。
高中：考入了邻近城市的一所重点高中——绿洲高中。在这里，她结识了许多志同道合的朋友，共同探讨文学创作。高中期间，她的小说首次在地方文学杂志上发表。
大学时期
本科：陈晓琳考入了位于首都的“国立文学学院”，主修中文文学。在大学期间，她加入了学校的文学创作社团，并担任社长。她的小说作品开始在全国性的文学比赛中获得认可。
硕士：继续在“国立文学学院”攻读文学创作硕士课程。在此期间，她出版了自己的第一部短篇小说集《青林故事》，并受到读者的喜爱。
职业生涯
编辑：毕业后，陈晓琳加入了一家名为“蓝天出版社”的知名出版社，成为了一名编辑。她在工作中结识了许多知名的作家，并从他们身上学到了很多宝贵的经验。
自由撰稿人：几年后，陈晓琳决定成为一名自由撰稿人，专注于自己的写作事业。她开始在各大文学期刊上发表作品，并逐渐积累了一批忠实读者。
成就与荣誉
随着时间的推移，陈晓琳的作品逐渐受到了更广泛的认可。她的第二本小说集《流年的风景》获得了国家级文学奖项，成为当年的畅销书之一。此外，她还参与了一些文学节和作家交流活动，与来自世界各地的作家进行了深入交流。

社会贡献
除了个人成就外，陈晓琳也十分注重回馈社会。她定期到川沙镇的学校进行文学讲座，鼓励学生们多读书、多写作，并设立了一个以自己名字命名的文学基金，用于支持年轻作家的创作。
"""
# 调用与配置向量模型
emb_model_path = "/root/project/My_projects/lang__chain/RAG_embedding_model/AI-ModelScope/bge-large-zh-v1___5"
emb_model_kwargs = {"device":"cuda"}
encode_kwargs = {"normalize_embeddings":True} ## set True to compute cosine similarity
emb_model = HuggingFaceBgeEmbeddings(
    model_name = emb_model_path,
    model_kwargs = emb_model_kwargs,
    encode_kwargs = encode_kwargs,
    query_instruction = "为这个句子生成表示以用于检索相关文章" 
)
# 初始化文本并分段
CV = [Document(person)] # Pass page_content in as positional or named arg
spliter = RecursiveCharacterTextSplitter(chunk_overlap=20, chunk_size=100)
splitted_cv = spliter.split_documents(CV)
# 创建RAG向量库
vector_repository = Chroma.from_documents(splitted_cv, emb_model) # Create a Chroma vectorstore from a list of documents.
# vector_repository = vector_repository.as_retriever().bind(k=2)  # 用了索引器之后才是Runnable的，这是第一种方法，不常用
# print(vector_repository)
vector_repository = RunnableLambda(vector_repository.similarity_search_with_relevance_scores).bind(k=2) ## 第二种方法实现RAG称为Runnable， 并且传入的是一个方法指针
# print(vector_repository)
# rt = vector_repository.similarity_search_with_relevance_scores("徐怡雯是男的还是女的", k=3)
# print(rt)


#### 创建提示语
#### 第一次回答问题
# 创建提示语 写法1
sys_template = SystemMessagePromptTemplate.from_template("你是一个专业的人工智能助手帮助客户解决问题，最后再进行英文翻译")
human_template = HumanMessagePromptTemplate.from_template("参考这个内容{ref}， 回答这个问题{content}")
prompt_template = ChatPromptTemplate.from_messages([
    (sys_template),
    (human_template)
])
value_input = {
    "content" : RunnablePassthrough(),
    "ref" : vector_repository}
# rt = prompt_template.invoke(value_input)
# print(rt)
# 创建输出解析器
str_parser = StrOutputParser()
# 创建链
chain = value_input | prompt_template | model | str_parser
res = chain.invoke("徐怡雯的家乡在哪里，上的什么学校")
print(res)


#### 创建提示语 写法2
#### 第二次回答问题
# 创建同一session_id的字典
session_store = {}
def get_session_history(session_id):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# 创建工具
tools_list = [weather, ecomomy, ]
model_with_tools = model.bind_tools(tools_list)  ## 嵌入工具

# 创建提示词模板
prompt_template = ChatPromptTemplate([
    ("system","你是一个专业的人工智能助手帮助客户解决问题，最后再进行{language}翻译"),
    MessagesPlaceholder(variable_name="history_contents"),
    ("user","{content}")  ##NotImplementedError: Unsupported message type: <class 'list'> 最好写元组
])
# rt = prompt_template.invoke({"language":"英文", "content":"徐怡雯的家乡在哪里，上的什么学校，家乡的天气如何？"})
# print(rt)

# 工具回答添加到原message中去
tools_ans = model_with_tools.invoke(prompt_template)
if tools_ans.tool_calls is not None or len(tools_ans.tool_calls) > 0:
    for _tool in tools_ans:
        _function = eval(_tool["name"])
        _tool_res = _function.invoke(_tool["args"])
        prompt_template.append(ToolMessage(_tool_res))
        
_chain = prompt_template | model | str_parser
chain_with_memory = RunnableWithMessageHistory(
    runnable = _chain,
    get_session_history = get_session_history,
    input_messages_key = "content",
    history_messages_key = "history_contents"
)

result = chain_with_memory.invoke({"language":"英文", "content":"上海的经济和天气如何？"}, config={"configurable":{"session_id":"1"}})
print(result)








