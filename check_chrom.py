from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. Load embedding ===
embedding_function = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

# === 2. Load Chroma database (ganti path sesuai folder kamu) ===
CHROMA_PATH = "chroma"   # contoh folder penyimpanan
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)

# === 3. Cek isi database ===
collection = db._collection  # ambil koleksi internal
print("📚 Jumlah dokumen di Chroma:", collection.count())

# === 4. Tampilkan ID dan metadata ===
docs = collection.get()
print("\n🆔 Daftar ID dokumen:")
for i, doc_id in enumerate(docs['ids'][:5]):  # tampilkan 5 pertama
    print(f"{i+1}. {doc_id}")

print("\n📑 Metadata contoh:")
for i, meta in enumerate(docs['metadatas'][:3]):  # tampilkan 3 metadata
    print(f"Metadata {i+1}: {meta}")

print("\n📄 Contoh isi teks:")
for i, text in enumerate(docs['documents'][:3]):  # tampilkan 3 teks pertama
    print(f"Chunk {i+1}: {text[:300]}...\n")  # tampilkan 300 karakter pertama

# === 5. (Opsional) Lakukan similarity search ===
query = "Apa kepanjangan dari BPUPKI?"
results = db.similarity_search(query, k=3)
print("\n🔍 Hasil pencarian similarity:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r.page_content[:300]}...\n(Sumber: {r.metadata})")
