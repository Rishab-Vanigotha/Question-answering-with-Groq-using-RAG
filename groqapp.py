# import os
# from dotenv import load_dotenv
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import time

## Input the Groq API Key
api_key=st.st.sidebar.text_input("Enter your Groq API key:",type="password")
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm_model = ChatGroq(model = "Llama3-8b-8192",api_key=groq_api_key)

prompt= ChatPromptTemplate.from_template(
    '''Answer the question based on the provided context only
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    '''
)

st.title("Question answering with Groq using RAG")

input_file = st.file_uploader("Please upload the file", type='pdf')

if input_file is not None:
    with open(input_file.name, mode='wb') as temp_file:
        temp_file.write(input_file.getvalue())

def create_vector_embeddings():
    if 'vectors' not in st.session_state:
        # st.session_state.embeddings = OllamaEmbeddings(model="gemma2:2b")
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFLoader(input_file.name) ## Data Ingestion
        st.session_state.documents = st.session_state.loader.load() ## Loading the Document after ingestion
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

query = st.text_input("Enter your query")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Your Vector Database is ready")

if query:
   doc_chain = create_stuff_documents_chain(llm_model, prompt=prompt) 
   retriever = st.session_state.vectors.as_retriever()
   rag_chain = create_retrieval_chain(retriever, doc_chain)

   start = time.process_time()
   response = rag_chain.invoke({'input':query})
   print("response time = ",time.process_time()-start)

   st.write(response['answer'])

   ##with a streamlit expander

   with st.expander('Document similarity search'):
       for i,doc in enumerate(response['context']):
           st.write(doc.page_content)
           st.write('------------------------')
