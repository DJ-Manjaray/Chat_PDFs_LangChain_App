from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain, LLMChain
from langchain.llms import HuggingFaceHub


class ChatPDFs:
    @staticmethod
    def get_pdf_text(pdf_docs):
        txt = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                txt += page.extract_text()
        return txt

    @staticmethod
    def get_txt_chunks(text):
        txt_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=205,
            length_function=len
        )
        chunks = txt_splitter.split_text(text)
        return chunks

    @staticmethod
    def get_vector_db(text_chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        vector_db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_db

    @staticmethod
    def get_conversation_chain(vector_db):
        repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
        llms = ChatOpenAI()
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
        )
        # llm_chain = LLMChain(prompt=prompt, llm=llm)

        # print(llm_chain.run(question))
        retriever = vector_db.as_retriever()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                                   retriever=retriever,
                                                                   memory=memory)
        return conversation_chain
