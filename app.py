import streamlit as st
from dotenv import load_dotenv
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_embeddings
from src.prompt import system_prompt

# -------------------------
# Load environment
# -------------------------

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

# Center title
st.markdown("<h1 style='text-align:center;'>🩺 Medical RAG Chatbot</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 New Chat"):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Connect Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

embeddings = download_embeddings()

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# -------------------------
# Memory
# -------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Chat history for UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# LLM
# -------------------------
chatmodel = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# RAG chain
# -------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatmodel,
    retriever=retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)
#-------------------------
# safety measure to 
# -------------------------
rewrite_prompt = """
You are a medical safety filter.

Your job is to rewrite the user question into a fully standalone medical question.

Rules:
- If the question depends on previous context (like "it", "this", "that", "its"),
  respond EXACTLY: NEED CLARIFICATION
- Do NOT guess missing information
- Do NOT invent a disease
- Only rewrite if the question is self-contained

User question: {question}
"""

def rewrite_question(question):
    rewritten = chatmodel.invoke(
        rewrite_prompt.format(question=question)
    ).content.strip()

    return rewritten


# -------------------------
# Replay old messages
# -------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -------------------------
# Chat input
# -------------------------
user_question = st.chat_input("Ask a medical question...")

if user_question:

    # Show user message
    st.chat_message("user").write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Rewrite for safety
    has_history = len(st.session_state.memory.chat_memory.messages) > 0
    if not has_history:
        rewritten = rewrite_question(user_question)
    else:
        rewritten = user_question

    # Debug (optional)
    print("Rewritten:", rewritten)

    # If unclear → stop
    if rewritten == "NEED CLARIFICATION":
        clarification = "Please clarify your question so I can help."
        st.chat_message("assistant").write(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
    
    else:
        # Use rewritten question for RAG
        docs = docsearch.similarity_search_with_score(rewritten, k=3)
        best_score = docs[0][1]
        if best_score < 0.6:  # confidence threshold
            st.chat_message("assistant").write("I'm not confident about the answer. Please consult a medical professional.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm not confident about the answer. Please consult a medical professional."})
        else:
            response = rag_chain({"question": rewritten})
            answer = response["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
