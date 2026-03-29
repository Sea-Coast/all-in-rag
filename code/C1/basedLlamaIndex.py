import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()

# 配置大模型
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)
# 设置嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 进行文档加载
documents = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 💡 Document 对象结构：
#   - page_content: 原始文本内容
#   - metadata: {'file_name': '...', 'file_path': '...', ...}


# 🗂️ 构建向量索引：文档 → 嵌入 → 向量存储
index = VectorStoreIndex.from_documents(documents)
# 🔍 内部流程：
#   1. 文档自动分块（默认 1024 token/块，重叠 20）
#   2. 调用 embed_model 将每块文本转为向量
#   3. 向量 + 原始文本 + 元数据 存入内存向量库（默认 InMemoryVectorStore）


# 🔍 创建查询引擎：检索 + 生成的流水线
query_engine = index.as_query_engine()

# 🔧 调试：查看查询引擎使用的提示词模板
print(query_engine.get_prompts())


print(query_engine.query("文中举了哪些例子?"))
