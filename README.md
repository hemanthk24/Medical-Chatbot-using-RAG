# 🩺 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) medical assistant built with LangChain, Pinecone, OpenAI, and Streamlit.  
The chatbot answers medical questions using a verified medical book as its knowledge base, with safety guardrails to reduce hallucinations.

🔗 **Live App:**  
https://medical-chatbot-using-rag-243.streamlit.app/

---

## 🚀 Features

- 📚 RAG pipeline over medical PDF knowledge base
- 🔍 Pinecone vector search with similarity scoring
- 🧠 Conversational memory (context-aware follow-up questions)
- 🛡 Safety rewrite filter to prevent ambiguous queries
- ❌ Confidence threshold to avoid hallucinated answers
- 💬 Streamlit chat UI with persistent session history
- 🔄 New chat reset per session
- 🔐 Secure API handling via Streamlit secrets
- ⚠ Medical disclaimer for safe usage

---

## 🧠 How It Works

1. User asks a medical question
2. Safety filter rewrites unclear queries
3. Pinecone retrieves the most relevant documents
4. Similarity score is checked for confidence
5. LLM answers only using retrieved context
6. Conversation memory maintains chat history

This architecture prevents the model from inventing medical information and ensures answers come from trusted documents.

---

## 🏗 Architecture

```
User → Safety Filter → Pinecone Retrieval → LLM → Response
           ↑                 ↓
      Conversation Memory (LangChain)
```

Technologies used:

- LangChain (RAG + memory)
- OpenAI GPT model
- Pinecone vector database
- HuggingFace embeddings
- Streamlit UI
- Python

---

## 📂 Project Structure

```
medical-chatbot-using-rag/
│
├── app.py                # Streamlit app
├── requirements.txt
├── data/
│   └── Medical_book.pdf
│
├── src/
│   ├── helper.py         # PDF loading + embeddings
│   ├── store_index.py    # Pinecone indexing
│   └── prompt.py         # system prompt
│
└── README.md
```

---

## ⚙ Installation (Local)

Clone repo:

```bash
git clone https://github.com/your-username/medical-chatbot-using-rag.git
cd medical-chatbot-using-rag
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` file:

```
PINECONE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

Run app:

```bash
streamlit run app.py
```

---

## 🔐 Secrets (Production)

In Streamlit Cloud:

```
PINECONE_API_KEY = ...
OPENAI_API_KEY = ...
```

Never commit API keys to GitHub.

---

## 🧪 Example Questions

- What is diabetes?
- What are its symptoms?
- Can it be dangerous?
- How is acne treated?
- What causes high blood pressure?

---

## ⚠ Disclaimer

This chatbot provides educational medical information only.  
It is **not a substitute for professional medical advice**.  
Always consult a qualified healthcare provider.

---

## 📈 Future Improvements

- Citation sources in answers
- Evaluation metrics dashboard
- Redis persistent memory
- User authentication
- Streaming responses
- Document highlighting

---

## 👨‍💻 Author

Built as a RAG + LLM safety engineering project.

If you like this project ⭐ the repo!
