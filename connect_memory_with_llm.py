import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from tabulate import tabulate

# setting up LLM(Mistral with Huggingface)
#HF_TOKEN=os.environ.get("HF_TOKEN")
HF_TOKEN="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"
#HF_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
HF_REPO_ID="HuggingFaceH4/zephyr-7b-beta"

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

#connecting LLM with FAISS and creating chain
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

def set_custom_prompt():
    return ChatPromptTemplate.from_messages([("system","You are a helpful medical assistant. Answer directly based on context."),
                                              ("human",CUSTOM_PROMPT_TEMPLATE)
                                              ])
     

# loading Database
DB_FAISS_PATH="vectorstores/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

# creating QA chain
chat_model=load_llm(HF_REPO_ID)
combine_docs_chain=create_stuff_documents_chain(chat_model,set_custom_prompt())
qa_chain=create_retrieval_chain(db.as_retriever(search_kwargs={'k':5}),combine_docs_chain)

# invoking with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'input':user_query})
print("\n" + "="*80)
print("MEDICAL ASSISTANT RESPONSE:")
print("-" * 80)
print(response['answer'])
print("-" * 80)

source_data = []
for doc in response['context']:
    page = doc.metadata.get('page', 0) + 1
    file = os.path.basename(doc.metadata.get('source', 'Unknown'))
    snippet = doc.page_content[:75].replace('\n', ' ') + "..."
    source_data.append([f"Page {page}", file, snippet])

print("\nSOURCES RETRIEVED FROM ENCYCLOPEDIA:")
print(tabulate(source_data, headers=["Location", "Filename", "Content Snippet"], tablefmt="grid"))
print("="*80)
