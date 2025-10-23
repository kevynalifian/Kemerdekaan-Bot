# Kemerdekaan-Bot
### Introduction
Perkenalkan saya Kevyn Alifian Hernanda Wibowo izin memperkenalkan "📖 Kemerdekaan Bot" yang dibuat untuk memenuhi test data science dari NoLimit. 📖 Kemerdekaan Bot adalah Chatbot berbasis RAG tentang Sejarah Proklamasi Kemerdekaan Indonesia, yang dibangun menggunakan google/embeddinggemma-300m, google/gemma-3-1b-it, dan ChromaDB.

### Dasar pemilihan model
1. google/embeddinggemma-300m
- Ringan & Efisien → Model embedding berukuran kecil (300M parameter) sehingga cepat dipakai untuk menghasilkan representasi vektor teks.
- Kualitas Representasi Baik → Mampu menghasilkan embedding yang cukup bagus untuk semantic search, clustering, dan retrieval, meskipun ukurannya tidak besar.
- Hemat Resource → Bisa jalan di GPU menengah (misalnya T4 di Colab), tidak perlu hardware super besar.

2. google/gemma-3-1b-it
- Ukuran kecil (1B parameter) → seimbang antara kecepatan dan kemampuan reasoning. Cocok untuk chatbot yang dijalankan lokal tanpa GPU besar.
- Teroptimasi untuk bahasa Indonesia dan multibahasa (karena dilatih dengan dataset global Google).

3. Elasticsearch
- Mudah diintegrasikan dengan LangChain & HuggingFace → cocok untuk pipeline RAG modern.
- Cepat dan ringan → bisa berjalan lokal (tanpa server) tapi mendukung query vektor jutaan entri.
- Persisten & portable → data bisa disimpan di direktori dan digunakan ulang tanpa re-indexing.

### Dataset
Dataset yang digunakan berasal dari https://www.ksap.org/sap/wp-content/uploads/2020/07/MAJALAH-MAYA-KSAP-1-AGUSTUS-2020.pdf

### Set up Instructions:
1. Install library yang tersedia pada requirements.txt, dengan menjalankan kode "pip install -r requirements.txt".
2. Login ke akun hugging face http://huggingface.co/.
3. Membuat access token di https://huggingface.co/settings/tokens , dan copy access tokennya.
4. Login hugging face dari terminal IDE, lalu paste access tokennya.
5. Kemudian jalankan file kemerdekaan_bot.py, cukup menggunakan kode "streamlit run kemerdekaan_bot.py".
6. Aplikasi 📖 Kemerdekaan Bot siap digunakan.

### Cara penggunaan 📖 Kemerdekaan Bot:
1. Pengguna memasukkan pertanyaan pada kolom yang tersedia.
2. Klik button "Cari Jawaban", kemudian tunggu hingga jawaban ditampilkan.
3. Jika sudah, 📖 Kemerdekaan Bot akan menampilkan jawaban dari pertanyaannya, konteks chunk, dan asal konteks chunk dari halaman berapa pada data .pdf yang digunakan

### Flowchart
<img width="2826" height="2521" alt="Flowchart" src="https://github.com/user-attachments/assets/b0a86990-8f84-472e-83d7-f7491fc39b8e" />
