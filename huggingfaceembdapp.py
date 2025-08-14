import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key,model = 'Llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the given questions in the provided context only
    provide the most accurate answer based on the context
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.document = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.document,st.session_state.embeddings)

st.title("Q&A LLM using HF and GROQ AI")

user_input = st.text_input("Enter the question")
if st.button("Vector Embeddings"):
    create_vector_embeddings()
    st.write("Vector db is ready")

if user_input:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriver = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriver,document_chain)

    response = retrieval_chain.invoke({'input':user_input})
    st.write(response['answer'])

    with st.expander("Document context"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------------')
