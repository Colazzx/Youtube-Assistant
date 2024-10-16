from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
from secret_key import openai_key

# Access the OpenAI API key from Streamlit Secrets
openai_key = st.secrets["OPENAI_API_KEY"]

# API Key for OpenAI
os.environ['OPENAI_API_KEY'] = openai_key

# Embeddings
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_link(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def response_from_query(db, query, k=4):
    # gpt-3.5-turbo-instruct has a 4096-token limit
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Adjust the content length to fit within the token limit
    max_token_length = 3500  # Leave room for prompt and response tokens
    if len(docs_page_content) > max_token_length:
        docs_page_content = docs_page_content[:max_token_length]

    llm = OpenAI(model="gpt-3.5-turbo-instruct-0914", temperature=0.1)

    prompt_template = """
              You are a helpful assistant that that can answer questions about youtube videos 
              based on the video's transcript.

              Answer the following question: {question}
              By searching the following video transcript: {docs}

              Only use the factual information from the transcript to answer the question.

              If you feel like you don't have enough information to answer the question, say "I don't know".

              Your answers should be verbose and detailed.
              """

    prompt = PromptTemplate(
        input_variables = ["question", "docs"],
        template = prompt_template
    )

    chain = LLMChain(
        llm = llm,
        prompt = prompt
    )

    response = chain.run(
        question = query,
        docs = docs_page_content
    )

    response = response.replace("\n", "")

    return response