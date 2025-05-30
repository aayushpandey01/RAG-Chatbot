This project is a Retrieval-Augmented Generation (RAG) Chatbot built using LangChain, Groq’s LLaMA 3 (70B) model, and Streamlit. It enables users to interactively query the contents of uploaded PDF documents. The chatbot uses advanced natural language processing and retrieval techniques to provide accurate, context-aware responses based on the documents.

Once users upload one or more PDFs, the chatbot extracts the text using PyPDF2, splits it into overlapping chunks using LangChain's CharacterTextSplitter, and generates embeddings via Groq's embedding model. These embeddings are stored in ChromaDB, a local vector database, which enables efficient semantic retrieval of relevant content when the user asks a question.

The retrieved chunks, along with the user's query and previous conversation history, are passed to the LLaMA 3 70B model hosted by Groq for generating responses. The application maintains conversation context using LangChain’s memory module, allowing for coherent follow-up questions.

The interface is built using Streamlit, providing a clean and interactive frontend. This project is ideal for creating personalized document assistants, customer support bots, or knowledge-based QA systems.

It supports multi-document processing, persistent memory, and delivers real-time answers—making it a powerful tool for document understanding and AI-driven conversation.



🚀 Features-


📄 Upload and process multiple PDF files

🔍 Split documents into manageable text chunks

🔗 Generate embeddings and store them using ChromaDB

🧠 Memory-enabled conversation with LLaMA 3 70B via Groq

💬 Persistent conversation history with contextual follow-ups

🖼️ Human and bot avatars for better UI (optional enhancement)



🛠️ Tech Stack-


Python

LangChain

Groq (LLaMA 3 70B) via langchain_groq

ChromaDB for vector storage

PyPDF2 for PDF parsing

Streamlit for the web UI

dotenv for secure API key handling



📂 How It Works-


User uploads one or more PDFs

Text is extracted and split into overlapping chunks

Each chunk is converted into an embedding via GroqEmbeddings

Embeddings are stored in ChromaDB

On question input, relevant chunks are retrieved and passed to LLaMA 3 via ChatGroq

The chatbot responds with contextual, accurate answers
