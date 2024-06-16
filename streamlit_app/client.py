from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama



def load_csv(file_path: str = 'cleaned_reviews.csv'):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data


def split_documents_into_chucks(data, chuck_size: int = 1000, chunk_overlap: int = 150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chuck_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

def get_hugging_face_embeddings(model_path: str = "sentence-transformers/all-MiniLM-l6-v2"):
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
    model_name=model_path, 
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs 
    )
    return embeddings


def create_documents_embeddings(docs: list, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    context_retriever = db.as_retriever(search_kwargs={"k": 100})
    return context_retriever


def get_model(model_name: str = "llama3"):
    model = Ollama(model=model_name, temperature=0.06)
    return model


def create_rag_with_history(model, retriever):
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    rag_chain_with_history = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain_with_history

def main():
    data = load_csv()
    docs = split_documents_into_chucks(data)
    embeddings = get_hugging_face_embeddings()
    c_retriever = create_documents_embeddings(docs, embeddings)

    model = get_model()

    rag_history = create_rag_with_history(model, c_retriever)

    return rag_history
