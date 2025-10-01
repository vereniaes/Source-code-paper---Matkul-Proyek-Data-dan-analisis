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
## Hasil Utama & Visualisasi
Pipeline ini berhasil mengidentifikasi dan memvisualisasikan pola perilaku anomali dalam jaringan.

**Peta Jaringan Interaksi**
Visualisasi ini memetakan seluruh jaringan interaksi. Ukuran titik merepresentasikan pengaruh akun (PageRank), sementara warna merepresentasikan komunitas (klaster) yang terdeteksi oleh Algoritma Louvain.

<img width="1570" height="1591" alt="image" src="https://github.com/user-attachments/assets/f2e8b2d7-1ac2-4428-a5a4-07dde8775851" />


Profil Perbandingan "Sidik Jari" Buzzer
Radar chart ini secara jelas menunjukkan perbedaan karakteristik antara akun yang diprediksi sebagai buzzer (merah) dengan pengguna biasa (hijau). Terlihat jelas bahwa akun buzzer memiliki skor tinggi pada `out_degree` (aktivitas menyebar) dan `narrative_similarity` (narasi seragam), namun rendah pada metrik interaksi organik.

<img width="881" height="742" alt="image" src="https://github.com/user-attachments/assets/036665a0-8ab8-4224-9287-b323e91bc24e" />


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

### 6. `phase6_inference_and_analysis.ipynb`

**Tujuan:** Menggunakan model terlatih untuk inferensi dan interpretasi hasil.

**Proses:**

- Melatih ulang model pada 100% data untuk mendapatkan prediksi terbaik.
- Mengurutkan semua akun berdasarkan probabilitas sebagai buzzer.
- Menganalisis dan memvisualisasikan karakteristik akun yang terprediksi sebagai buzzer.

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
git clone git@github.com:vereniaes/Source-code-paper---Matkul-Proyek-Data-dan-analisis.git
cd Source-code-paper---Matkul-Proyek-Data-dan-analisis
pip install -r requirements.txt
```

### 3. Menjalankan Notebook

Jalankan file Jupyter Notebook secara berurutan, karena output dari satu phase adalah input untuk phase berikutnya:

1. `phase1_preprocessing.ipynb`
2. `phase2_graph_and_sna.ipynb`
3. `phase3_feature_engineering.ipynb`
4. `phase4_heuristic_labeling.ipynb`
5. `phase5_model_training.ipynb`
6. `phase6_inference_dan_analysis.ipynb`

### **Kesimpulan Proyek**
Proyek ini berhasil membangun sebuah prototipe end-to-end yang fungsional untuk mendeteksi akun anomali. Poin-poin utamanya adalah:
- Metodologi Valid: Fusi fitur dari domain jaringan, konten, dan perilaku terbukti efektif.
- Tantangan Utama: Kualitas model sangat bergantung pada kuantitas dan kualitas data berlabel. Kurangnya sampel positif (is_buzzer=1) adalah batasan terbesar.
- Arah Pengembangan: Langkah selanjutnya adalah menerapkan pendekatan human-in-the-loop, di mana hasil prediksi model digunakan untuk membantu investigator manusia melakukan pelabelan lebih lanjut, yang kemudian dapat digunakan untuk melatih ulang model yang lebih kuat.
