import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 加载原始文档
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = TextLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
texts = text_splitter.split_documents(docs)


# 构建索引
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-zh-v1.5",
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True}
)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)

# 查询与检索
question = "文中举了那些例子"
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)


# 生成集成
# 构建提示词模版
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文，如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”
上下文：{context}
问题：{question}
回答："""
)

# 配置大语言模型
llm = ChatOpenAI(
    model = "glm-4.7-flash-free",
    temperature = 0.7,
    max_tokens = 2048,
    api_key = os.getenv("AIHUBMIX_API_KEY"),
    base_url = "https://aihubmix.com/v1"
)

# 调用LLM生成答案并输出
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
