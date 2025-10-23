import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

CHROMA_PATH = r"chroma"

# Prompt template
PROMPT_TEMPLATE = """
<start_of_turn>user
Jawablah pertanyaan di bawah ini hanya dengan mengutip dari konteks. 
Jika tidak ada jawaban di konteks, jawab: "Informasi tidak tersedia".

Konteks:
{context}

Pertanyaan:
{question}<end_of_turn>
<start_of_turn>model
"""

@st.cache_resource
def load_db():
    embedding_function = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m"
    )
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    return db


def rag_answer(query_text: str):
    db = load_db()
    results = db.similarity_search_with_score(query_text, k=3)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    generator = pipeline(
        task="text-generation",
        model="google/gemma-3-1b-it",
        device=0,
        do_sample=True,
        temperature=0.2,
    )

    llm = HuggingFacePipeline(pipeline=generator)
    response = llm.invoke(prompt)

    # Ekstraksi jawaban dari format turn
    if isinstance(response, str):
        try:
            answer_start = response.index("<start_of_turn>model") + len("<start_of_turn>model")
            answer = response[answer_start:].strip()
        except ValueError:
            answer = response
    else:
        answer = str(response)

    sources = []
    for doc, _ in results:
        src_meta = getattr(doc, "metadata", {})
        src = src_meta.get("source", "Unknown")
        page = src_meta.get("page", None)
        if page is not None:
            sources.append(f"{src} (halaman {page+1})")
        else:
            sources.append(src)

    chunks = [doc.page_content for doc, _ in results]
    return chunks, answer.strip(), sources

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="📖 Kemerdekaan Bot", page_icon="📚")
st.title("📖 Kemerdekaan Bot")
st.write("Chatbot tentang Sejarah Proklamasi Kemerdekaan Indonesia.")
st.write("Chatbot berbasis RAG ini dibangun menggunakan google/embeddinggemma-300m, gemma-3-1b-it, dan ChromaDB.")

user_query = st.text_input("Masukkan pertanyaan Anda:")

if st.button("Cari Jawaban") and user_query:
    with st.spinner("Sedang mencari jawaban..."):
        chunks, answer, sources = rag_answer(user_query)

    st.subheader("Jawaban")
    st.write(answer)

    with st.expander("Lihat Context pada Chunk"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}:**\n\n{chunk}")

    st.subheader("Sumber Dokumen untuk Chunk")
    for i, src in enumerate(sources, 1):

        st.write(f"{i}. {src}")
