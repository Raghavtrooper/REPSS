import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda

# Import os for os.getenv
import os # Added this import

# Import configuration from shared module
# Expecting OLLAMA_MODEL from shared.config
try:
    from shared.config import OLLAMA_MODEL
except ImportError:
    # Fallback if shared.config is not available or missing variables (for standalone testing)
    print("Warning: shared/config.py not found or incomplete. Using default values.")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2") # Default model if not from config


@st.cache_resource(show_spinner=False)
def get_rag_chain(_vectorstore):
    """
    Initializes and returns the RAG (Retrieval Augmented Generation) chain.
    The chain is designed to accept explicit 'context' in its invoke method.
    If 'context' is provided, it will be used directly; otherwise, the vectorstore's
    retriever will be used to fetch documents.
    """
    print(f"Initializing RAG chain with model: {OLLAMA_MODEL}")
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1) # Set a low temperature for more consistent output

    # Contextualize question prompt: used to rephrase the user's question
    # into a standalone question given the chat history.
    contextualize_q_system_prompt = """Given the conversation history and a new user question, formulate a standalone question that can be understood without the chat history. This is crucial for accurate information retrieval.
    - If the new question builds on previous turns, incorporate necessary details from the history.
    - If the new question is a completely new topic, return it as is.
    - Do NOT answer the question.
    - Ensure the standalone question is concise and precise for searching.

    Example 1:
    Chat History:
    Human: Who is John Doe?
    AI: John Doe is a Senior Software Engineer in the Engineering department.
    Human: What are his skills?
    Standalone Question: What are John Doe's skills?

    Example 2:
    Chat History:
    Human: Find employees with Python skills.
    AI: I found 15 Python developers.
    Human: How many have AWS experience?
    Standalone Question: How many Python developers have AWS experience?

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
    Use ONLY the following retrieved context to answer the question.
    - If the answer is not found in the context, clearly state "I don't have information on that specific detail based on the profiles I have." Do NOT make up answers.
    - Synthesize information from the provided context without repetition.
    - Provide relevant and detailed information.

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
