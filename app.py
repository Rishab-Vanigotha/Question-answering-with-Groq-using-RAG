import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

#Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&AChatbot Application"

#Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
    ("system","You are a helpful assistant. please respond to user queries"),
    ("user","Question: {question}")
    ]
)

def generate_response(question, model, temperature, max_token):
    llm_model = Ollama(model = model)
    output_parser = StrOutputParser()
    chain = prompt | llm_model | output_parser
    answer = chain.invoke({'question':question})
    return answer

##Title of the app
st.title("Enhanced Q&A chatbot with Ollama")

##Drop down to select the Ollama model
model = st.sidebar.selectbox("Select the AI model: ", ["gemma2:2b","llama3.2"])

temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Max_Token",min_value=50, max_value=300,value=150)

### Main interface for user input

st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(question=user_input, model = model, temperature=temperature, max_token=max_token)
    st.write(response)
else:
    st.write("please provide the query")