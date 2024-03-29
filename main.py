from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from llama_index.core.schema import TextNode
from langchain.storage import LocalFileStore
from langchain import hub
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_core .vectorstores import VectorStoreRetriever
from langchain.embeddings import CacheBackedEmbeddings
#os.environ["OPENAI_API_KEY"] =getpass.getpass("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")
#os.environ["OPENAI_API_KEY"] ="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"

os.environ['QIANFAN_AK'] = "OUkxyg8HHMlqazGXax5sJvsM"
os.environ['QIANFAN_SK'] = "cojenavz34hwQ754SsfcQXXBLx39KraM"

#创建文档加载对象，
loader = TextLoader(r'E:\算法实验\信息检索增强\news\corpus.txt',encoding='UTF-8')
doc=loader.load()

#创建文本拆分器
text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_spliter.split_documents(doc)

#创建向量化模型
embedder=QianfanEmbeddingsEndpoint(model='bge_large_zh',endpoint='bge_large_zh')
store = LocalFileStore("./cache/")
cache_embedder=CacheBackedEmbeddings.from_bytes_store(embedder, store,namespace=embedder.model)


#创建向量存储器
db=Chroma.from_documents(docs, cache_embedder)

retrieval=VectorStoreRetriever(vectorstore=db)
prompt=hub.pull("rlm/rag-prompt")
chat = QianfanChatEndpoint(model="ERNIE-Bot-4")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retrieval | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

#
res=rag_chain.invoke("十八世纪，中国的人口发生了什么变化?")
print(res)