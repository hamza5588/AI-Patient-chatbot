from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from embeddings import chat_model, embeddings
from langchain.chains import create_history_aware_retriever

# Define your prompt templates
prompt_template = ChatPromptTemplate.from_template(
    """You are an AI chatbot simulating a patient for paramedic students practicing clinical assessments.
    Respond accurately using the predefined question:answer pairs for clinical-related inquiries.
    If no predefined answer exists, provide a general conversational response while staying in character as a patient.
    
    <context>
    {context}
    </context>
    
    Question: {input}"""
)

system_prompt = (
    "You are simulating a real patient for paramedic students. "
    "You will respond using predefined question-answer pairs where possible. "
    "If no predefined answer exists, respond realistically as a patient would, "
    "but never break character or reveal you are an AI. Stay within the clinical context at all times."
    "{context}"
)

# Load your vectorstore
vector_store = FAISS.load_local(r"content\faissindexupdate", embeddings, allow_dangerous_deserialization=True)

# Create document chain
document_chain = create_stuff_documents_chain(chat_model, prompt_template)

# Create retriever and retrieval chain
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, 
    ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question..."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
)

# Create final question-answering chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

# Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)