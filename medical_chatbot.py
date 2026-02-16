import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS


# DB_FAISS_PATH="vectorstore/db_faiss"
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH=os.path.join(BASE_DIR,'vectorstores','db_faiss')

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db


HF_TOKEN="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"
#HF_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
HF_REPO_ID="HuggingFaceH4/zephyr-7b-beta"


def set_custom_prompt(custom_prompt_template):
    return ChatPromptTemplate.from_messages([("system","You are a helpful medical assistant. Answer directly based on context."),
                                              ("human",custom_prompt_template)
                                              ])
def load_llm(hf_repo_id):
    if not HF_TOKEN:
        raise ValueError("HF_Token environment variable is not set!")
    llm=HuggingFaceEndpoint(
        repo_id=hf_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)

def main():
    st.title("Ask MediBot!")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    prompt=st.chat_input("Describe your medical problem here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        CUSTOM_PROMPT_TEMPLATE="""
        As a professional medical assistant, provide a clear and concise summary based ONLY on the context below. 
        Provide a single, focused answer to the user's question. Do not include information about related topics
        found in the context unless they directly explain the user's query.

        CONTEXT:
        {context}

        USER QUESTION: 
        {input}

        INSTRUCTIONS:
        - Do not repeat yourself.
        - Use bullet points for clarity if there are multiple steps.
        - If the context doesn't contain the answer, state that you don't know.
        - End with a disclaimer: "Consult a doctor for personalized medical advice."

        ANSWER:
        """

        HF_REPO_ID="HuggingFaceH4/zephyr-7b-beta"
        HF_TOKEN="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"
       
        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector store")

            chat_model=load_llm(HF_REPO_ID)
            combine_docs_chain=create_stuff_documents_chain(chat_model,set_custom_prompt(CUSTOM_PROMPT_TEMPLATE))
            qa_chain=create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k':5}),combine_docs_chain)
            
            response=qa_chain.invoke({'input':prompt})
            result=response['answer']
            #source_documents=response['context']
            result_to_show=result
        
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant','content':result_to_show})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__=="__main__":
    main()
