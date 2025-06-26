from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda

# Import os for os.getenv
import os

# Import configuration from shared module
try:
    from shared.config import OLLAMA_MODEL
except ImportError:
    print("Warning: shared/config.py not found or incomplete. Using default values.")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")


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
    AI: John Doe is a Senior Software Engineer.
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
    Your goal is to help identify suitable employees based on their skills and experience.
    Use the following retrieved context to answer the question.
    The context will contain information such as name, email, phone, location, objective, skills,
    qualifications summary, experience summary, and whether they have a photo.
    - If the answer is not found in the context, clearly state "I don't have information on that specific detail based on the profiles I have." Do NOT make up answers.
    - Synthesize information from the provided context without repetition.
    - Provide relevant and detailed information.

    When asked to judge or compare proficiency for a role (e.g., "best for architecture planning" or "most charismatic"),
    you should:
    1.  **Prioritize direct matches:** Look for explicit mentions of the role or related titles (e.g., "Architect," "System Designer," "Leadership," "Communication") within the 'experience_summary' and 'objective' fields in the provided employee profiles.
    2.  **Infer from skills and experience:** If direct matches are not found, analyze the listed skills, 'qualifications_summary', and 'experience_summary' in the provided context and use your general knowledge about what skills and attributes are typically required or highly beneficial for that role or trait.
        * For example, for "software architecture planning," consider skills like "System Design," "Cloud Architecture," "Microservices," "Scalability," "Enterprise Integration Patterns," "UML," "design patterns," as well as strong foundational programming languages if explicitly mentioned alongside relevant design principles.
        * For "charismatic," look for indicators within 'experience_summary' or 'objective' such as "Project Management" (implying communication and coordination), "Team Lead" experience, or even a broad range of skills that suggest adaptability and interaction.
    3.  **Consider information in summaries:** Factor in details found in `experience_summary` and `qualifications_summary` to gauge overall suitability.
    4.  **Acknowledge limitations:** If, even after inference, you cannot confidently identify a "best" candidate or if the information is insufficient, state that your assessment is based on the available data and that further details on project roles or specific contributions would be needed for a definitive judgment.
    5.  **Summarize findings:** Present the profiles that seem most relevant based on your analysis, briefly explaining *why* you believe their skills are applicable to the requested role or trait.
    6.  **Avoid Repetition (if diverse options exist):** If you are asked similar questions repeatedly and have already suggested certain profiles, try to identify and suggest other suitable, *unmentioned* profiles from the *provided context*, if they exist and are relevant. Do not force recommendations if no other suitable candidates are present in the context.

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
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- Modified Retrieval Logic ---
    def custom_retriever_logic(input_dict):
        if "context" in input_dict and input_dict["context"] is not None:
            return input_dict["context"]
        else:
            print("  No explicit 'context' provided. Using _vectorstore.as_retriever().")
            return _vectorstore.as_retriever().invoke(input_dict["input"])

    custom_retriever_runnable = RunnableLambda(custom_retriever_logic)

    # Create the retrieval chain with our custom runnable retriever
    retrieval_chain = create_retrieval_chain(custom_retriever_runnable, document_chain)

    print("RAG chain initialized successfully.")
    return retrieval_chain
