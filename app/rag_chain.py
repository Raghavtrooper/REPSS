import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda # ADDED THIS IMPORT for RunnableLambda

# Import configuration from shared module
from shared.config import OLLAMA_MODEL

@st.cache_resource(show_spinner=False)
def get_rag_chain(_vectorstore): # Changed 'vectorstore' to '_vectorstore' for caching
    """
    Initializes and returns the RAG (Retrieval Augmented Generation) chain.
    The chain is designed to accept explicit 'context' in its invoke method.
    If 'context' is provided, it will be used directly; otherwise, the vectorstore's
    retriever will be used to fetch documents.
    """
    print(f"Initializing RAG chain with model: {OLLAMA_MODEL}")
    llm = Ollama(model=OLLAMA_MODEL)

    # Contextualize question prompt: used to rephrase the user's question
    # into a standalone question given the chat history.
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just rephrase it if necessary and retain all original keywords.
    If no relevant chat history is provided, just return the original question."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Answer generation prompt: used to generate the final answer from retrieved documents.
    qa_system_prompt = """You are an AI assistant for questions about employee profiles.
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide detailed information based on the context.

    Context:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Define the document combining chain
    # This chain will take retrieved documents and format them into the 'context' variable
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- Modified Retrieval Logic ---
    # Instead of always using _vectorstore.as_retriever(), we will check if 'context' is
    # explicitly provided in the invoke call from main_app.py.
    # If 'context' is provided, we use it directly; otherwise, we use the _vectorstore's retriever.
    
    # Create a runnable from our custom retriever logic
    # This allows create_retrieval_chain to still work by treating it as a Runnable
    def custom_retriever_logic(input_dict):
        # If 'context' is explicitly passed from the caller (main_app.py), use that
        if "context" in input_dict and input_dict["context"] is not None:
            # Ensure it's a list of Documents or convert if needed (main_app.py already converts)
            return input_dict["context"] 
        else:
            # Otherwise, use the standard _vectorstore retriever
            print("  No explicit 'context' provided. Using _vectorstore.as_retriever().")
            # The input for _vectorstore.as_retriever() is expected to be the query from 'input'
            return _vectorstore.as_retriever().invoke(input_dict["input"]) 
    
    # Wrap the custom logic in a RunnableLambda to make it a Runnable
    custom_retriever_runnable = RunnableLambda(custom_retriever_logic)
    
    # Create the retrieval chain with our custom runnable retriever
    retrieval_chain = create_retrieval_chain(custom_retriever_runnable, document_chain)

    print("RAG chain initialized successfully.")
    return retrieval_chain
