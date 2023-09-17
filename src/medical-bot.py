from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pinecone
import os
from dotenv import load_dotenv

# Xác định đường dẫn đến tệp .env bên ngoài thư mục src
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

load_dotenv(dotenv_path)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')


os.environ["OPENAI_API_KEY"] = ''

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = OpenAI(model_name="gpt-3.5-turbo",max_new_tokens = 512,
        temperature = 0.5)
    return llm
     

#loading vector db
def load_db():
    embeddings = OpenAIEmbeddings()
    db = pinecone.init(      
	    api_key=PINECONE_API_KEY,      
	    environment=PINECONE_ENV      
    )
    db = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    return db

def bot():
    
    db = load_db()
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    bot = retrieval_qa_chain(llm, qa_prompt, db)

    return bot

#output function
def final_result(query):
    qa_result = bot()
    response = qa_result({'query': query})
    return response

    


