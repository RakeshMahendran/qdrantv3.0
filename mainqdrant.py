import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever

# Load environment variables
load_dotenv()

def setup_openai_api():
    """Setup OpenAI API Key"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    os.environ["OPENAI_API_KEY"] = openai_api_key

def define_llm():
    """Define the LLM"""
    return ChatOpenAI(model="gpt-4o-mini")

def retrieve_vectorstore():
    """Retrieve an existing vector store"""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        st.error("QDRANT_URL or QDRANT_API_KEY is not set in the .env file.")
        return None

    return QdrantVectorStore.from_existing_collection(
        url=url,
        api_key=api_key,
        collection_name="second-vector-db",
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    )

def create_vectorstore(documents):
    """Create a new vector store from documents"""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        st.error("QDRANT_URL or QDRANT_API_KEY is not set in the .env file.")
        return None

    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        url=url,
        api_key=api_key,
        collection_name="justcheck",
    )

def setup_retrievers(llm, vectorstore1, vectorstore2):
    """Set up history-aware retriever with two vector stores"""
    retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 12})
    retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 12})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.5, 0.5]
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or initialize chat session history"""
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def create_conversational_chain(rag_chain):
    """Create conversational chain with message history"""
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.create_documents([text])

def main():
    st.title("Conversational RAG App with PDF Ingestion")

    setup_openai_api()
    llm = define_llm()

    # Retrieve existing vector stores
    vectorstore1 = retrieve_vectorstore()
    if not vectorstore1:
        return

    # PDF Upload Section
    st.header("Upload a PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            # Split text into chunks
            documents = split_text_into_chunks(text)
            # Create a new vector store
            vectorstore2 = create_vectorstore(documents)

            if not vectorstore2:
                st.error("Failed to create vector store for PDF documents.")
                return

            st.success("PDF processed and added to vector store.")

            # Combine retrievers and create RAG chain
            rag_chain = setup_retrievers(llm, vectorstore1, vectorstore2)
            conversational_rag_chain = create_conversational_chain(rag_chain)

            # Question-Answer Section
            st.header("Ask a Question")
            user_input = st.text_area("Your question")
            if st.button("Ask"):
                with st.spinner("Generating response..."):
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default-session"}},
                    )
                    st.write(response)

if __name__ == "__main__":
    main()
