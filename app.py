import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from groq_embeddings import GroqEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import os
import hashlib
import uuid

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def generate_unique_dir(pdf_docs):
    file_names = "_".join([pdf.name for pdf in pdf_docs])
    unique_id = hashlib.sha256(file_names.encode()).hexdigest()
    persist_directory = os.path.join("chroma_dbs", unique_id)
    os.makedirs(persist_directory, exist_ok=True)
    return persist_directory

def get_vectorstore(text_chunks, persist_directory):
    embeddings = GroqEmbeddings()
    
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = vectorstore.as_retriever()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def handle_userinput(user_question):
    try:
        if st.session_state.conversation is None:
            st.error("Please upload and process your documents first!")
            return

        response = st.session_state.conversation.invoke({
            'question': user_question
        })

        st.session_state.chat_history = response['chat_history']

        st.write("### Answer")
        st.write(response['answer'])

    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("🤖 RAG Chatbot")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                persist_directory = generate_unique_dir(pdf_docs)
                vectorstore = get_vectorstore(text_chunks, persist_directory)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")

if __name__ == '__main__':
    main()

    
