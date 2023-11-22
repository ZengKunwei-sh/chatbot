from glm import GLM
from vector_store import init_knowledge_vector_store
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class QA_Chatbot(object):
    model_name_or_path: str = 'E:\chatbot\chatglm2-6b'
    device: str = 'cpu'
    search_k: int = 3

    def __init__(self, folder_path, vs_path=None) -> None:
        self.model = GLM()
        self.model.load_model(self.model_name_or_path, self.device)
        self.vector_store = init_knowledge_vector_store(folder_path, vs_path)
        self.prompt = self._generate_prompt()
        chain_type_kwargs = {'prompt': self.prompt}
        self.memory = ConversationBufferMemory(memory_key='chat_history', 
                                          return_messages=True, 
                                          input_key='question', output_key='answer')
        self.QaChain = ConversationalRetrievalChain.from_llm(llm=self.model, 
                                            chain_type="stuff", 
                                            retriever=self.vector_store.as_retriever(search_kwargs={'k': self.search_k}), 
                                            combine_docs_chain_kwargs=chain_type_kwargs,
                                            return_source_documents=True,
                                            memory=self.memory)

    def _generate_prompt(self):
        prompt_template = """根据以下文档内容回答问题，若无法根据文档内容回答，则只说不知道。仅需直接回答问题本身，不需要说明依据。

        文档内容：{context}

        问题：{question}
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=['context', 'question']
        )
        return prompt
    
    def ask(self, query):
        response = self.QaChain({'question': query})
        answer = response['answer']
        source_text = []
        for i, source in enumerate(response['source_documents']):
            content = source.page_content
            source_doc = source.metadata['source']
            source_text.append(f'[{i+1}] {content} ({source_doc})')
        return '\n'.join([answer] + source_text)

if __name__ == '__main__':
    qa = QA_Chatbot('documents')
    for i in range(5):
        query = input('请开始提问:')
        print(qa.ask(query))