# Pemodelan Prediktif untuk Klasifikasi Akun Buzzer (SDG 16)

Sebuah Proyek Data Science untuk Menjaga Integritas Ruang Diskusi Publik dan Mendukung Institusi yang Akuntabel

---

## 📖 Deskripsi Proyek

Proyek ini adalah implementasi end-to-end dari pipeline data science untuk mengidentifikasi dan memprediksi akun "buzzer" atau bot di platform media sosial X (sebelumnya Twitter). Di era digital, maraknya disinformasi yang disebarkan oleh aktor terkoordinasi menjadi ancaman serius bagi demokrasi dan kepercayaan publik.

Dengan memanfaatkan teknik Social Network Analysis (SNA), Natural Language Processing (NLP), dan Predictive Modeling, proyek ini bertujuan untuk membangun sebuah model klasifikasi machine learning yang mampu mengenali akun-akun anomali berdasarkan "sidik jari" digital mereka: pola interaksi, keseragaman narasi, dan perilaku aktivitas.

---

## 🎯 Kontribusi untuk SDG 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh

Proyek ini secara langsung berkontribusi pada Target 16.10: Menjamin akses publik terhadap informasi dan melindungi kebebasan fundamental. Dengan menciptakan alat untuk mendeteksi polusi informasi, kita berupaya:

- **Melindungi Ruang Publik:** Membantu membersihkan ruang diskusi dari narasi artifisial.
- **Meningkatkan Akuntabilitas:** Memberikan data bagi institusi untuk memahami manipulasi opini publik.
- **Memperkuat Kepercayaan:** Mengembalikan kepercayaan masyarakat terhadap informasi yang otentik.

---

## ⚙️ Metodologi & Pipeline Proyek

Pendekatan utama proyek ini adalah fusi fitur multi-dimensi. Kami tidak hanya melihat apa yang dikatakan sebuah akun, tetapi bagaimana mereka mengatakannya, dengan siapa mereka berinteraksi, dan pola aktivitas mereka. Pipeline ini terbagi menjadi beberapa notebook Jupyter yang harus dijalankan secara berurutan.

### 1. `phase1_preprocessing.ipynb`

**Tujuan:** Membersihkan dan menggabungkan data mentah dari berbagai isu (RKUHP, DPR, dll.) yang diambil dari Twitter.

**Proses:**

- Memuat dataset JSON dari berbagai topik.
- Menangani nama kolom yang tidak konsisten dan membersihkan data.
- Melakukan normalisasi teks (menghapus URL, mention, hashtag, dan karakter non-alfabet).
- Mengekstrak informasi `reply_to_user` dari teks mentah.
- Menggabungkan semua data menjadi satu `master_dataset_cleaned.csv`.

### 2. `phase2_graph_and_sna.ipynb`

**Tujuan:** Mengubah data interaksi tabular menjadi model graf jaringan dan mengekstrak fitur topologi.

**Proses:**

- Membangun graf berarah (Directed Graph) menggunakan NetworkX, di mana akun adalah node dan balasan adalah edge.
- Menghitung metrik-metrik SNA kunci untuk setiap akun.
- Mengidentifikasi klaster atau "gelembung" percakapan menggunakan Algoritma Louvain Community Detection.
- Menyimpan hasilnya ke `network_features.csv`.

### 3. `phase3_feature_engineering.ipynb`

**Tujuan:** Memperkaya data dengan fitur-fitur baru dari konten dan perilaku.

**Proses:**

- **Fitur Konten:** Menghitung skor `narrative_similarity` untuk setiap akun menggunakan TF-IDF dan Cosine Similarity, mengukur seberapa seragam narasi sebuah akun dibandingkan dengan komunitasnya.
- **Fitur Perilaku:** Menghitung `tweet_frequency` (rata-rata tweet per hari) dan `reply_ratio` (rasio balasan terhadap total aktivitas).
- Menyimpan hasilnya ke `final_features.csv`.

### 4. `phase4_heuristic_labeling.ipynb`

**Tujuan:** Membuat label data latih (`is_buzzer`) karena tidak adanya ground truth.

**Proses:**

- Menerapkan Pelabelan Heuristik berbasis aturan yang ketat berdasarkan kombinasi fitur ekstrem yang mengindikasikan perilaku anomali.
- **Aturan:** `in_degree = 0`, `betweenness = 0`, `narrative_similarity > 75%`, `out_degree > 75%`.
- Menyimpan dataset akhir yang siap dilatih ke `labeled_dataset.csv`.

### 5. `phase5_model_training.ipynb`

**Tujuan:** Melatih dan mengevaluasi model klasifikasi prediktif.

**Proses:**

- Membagi dataset menggunakan Stratified Train-Test Split untuk menjaga proporsi kelas yang sangat tidak seimbang.
- Melatih model XGBoost Classifier dengan parameter `scale_pos_weight` untuk menangani ketidakseimbangan kelas.
- Mengevaluasi model menggunakan metrik yang relevan: Confusion Matrix, Classification Report (Recall, F1-Score), dan Precision-Recall Curve (AUC-PR).

---

## 🛠️ Algoritma & Teknologi yang Digunakan

| Kategori            | Algoritma / Teknik         | Phase Digunakan | Tujuan                                               |
| ------------------- | -------------------------- | --------------- | ---------------------------------------------------- |
| Analisis Jaringan   | PageRank, Betweenness      | Phase 2         | Mengukur pengaruh dan peran akun dalam jaringan      |
| Deteksi Komunitas   | Louvain Method             | Phase 2         | Mengelompokkan akun ke dalam klaster interaksi       |
| NLP & Konten        | TF-IDF & Cosine Similarity | Phase 3         | Mengukur keseragaman narasi tanpa memahami makna     |
| Pelabelan Data      | Heuristic Labeling         | Phase 4         | Menciptakan label proksi saat ground truth tidak ada |
| Pemodelan Prediktif | Stratified Split, XGBoost  | Phase 5         | Melatih model klasifikasi pada data tidak seimbang   |

**Teknologi Utama:**  
Python, Pandas, NetworkX, Scikit-learn, XGBoost, Matplotlib

---

## 🚀 Cara Menjalankan Proyek

Untuk mereplikasi hasil dari proyek ini, ikuti langkah-langkah berikut:

### 1. Prasyarat

- Python 3.8 atau lebih tinggi
- Jupyter Notebook atau JupyterLab

### 2. Instalasi

Clone repositori ini dan instal semua library yang dibutuhkan. Disarankan untuk menggunakan virtual environment.

```bash
git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]
pip install -r requirements.txt
```

### 3. Menjalankan Notebook

Jalankan file Jupyter Notebook secara berurutan, karena output dari satu phase adalah input untuk phase berikutnya:

1. `phase1_preprocessing.ipynb`
2. `phase2_graph_and_sna.ipynb`
3. `phase3_feature_engineering.ipynb`
4. `phase4_heuristic_labeling.ipynb`
5. `phase5_model_training.ipynb`
