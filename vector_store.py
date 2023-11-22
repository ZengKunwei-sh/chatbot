from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def load_file(file_path):

    def preprocess(doc):
        doc.page_content = doc.page_content.strip('\n')
        return doc

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5, separators='\n')
    docs = text_splitter.split_documents(docs)
    return list(map(preprocess, docs))

def init_knowledge_vector_store(folder_path, vs_path=None):
    if not os.path.exists(folder_path):
        print(f'{folder_path} 路径不存在')
        return None
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        docs.extend(load_file(file_path))
        print(f'成功加载 {file}')
    if not docs:
        print('该路径下无文件')
        return None
    cache_folder = './embedding_model/GanymedeNil_text2vec-large-chinese'
    #"GanymedeNil/text2vec-large-chinese"
    embeddings = HuggingFaceEmbeddings(model_name=cache_folder,
                                    model_kwargs={'device': 'cpu'})
    if vs_path and os.path.exists(vs_path):
        vector_store = FAISS.load_local(vs_path, embeddings)
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
    print('向量库成功生成')
    if vs_path:
        vector_store.save_local(vs_path)
        print(f'向量库已保存至 {vs_path}')
    return vector_store

def search_similar_documents(query, vector_store, k, with_score=True):
    """
    # 使用 retriever 方法，但无法返回带得分的文档
    search_kwargs={'k': k}
    retriver = vector_store.as_retriever(search_type='mmr', 
                                         search_kwargs=search_kwargs)
    docs = retriver.get_relevant_documents(query)
    """
    if with_score:
        docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    else:
        docs = vector_store.similarity_search(query, k=k)
    return docs

if __name__ == '__main__':
    vector_store = init_knowledge_vector_store('documents')
    #docs = search_similar_documents('曾昆炜的星座', vector_store, 4, True)
    #print(docs)