### 固定大小分快

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("../../data/C2/txt/蜂医.txt")
# docs = loader.load()

# text_splitter = CharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap = 10
# )

# chunks = text_splitter.split_documents(docs)

# # print(chunks)

# print(f"文本被切分为{len(chunks)}个块。\n")

# print("---前五个块内容示例---")

# for i, chunk in enumerate(chunks[:5]):
#     print("="*60)
#     print(f'块{i+1}(长度：{len(chunk.page_content)}):"{chunk.page_content}"')

### 递归字符分块

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# loader = TextLoader("../../data/C2/txt/蜂医.txt")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     separators = ["\n\n", "\n", ".", ",", ""], # 分隔符优先级
#     chunk_size = 200,
#     chunk_overlap = 10
# )

# chunk = text_splitter.split_documents(docs)
# print(chunk)



### 语义分块

import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-zh-v1.5",
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

# 初始化SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type = "percentile"   # 断点识别方法
)

loader = TextLoader("../../data/C2/txt/蜂医.txt")
docunments = loader.load()

docs = text_splitter.split_documents(docunments)
print(docs)


### 基于文档结构的分块

