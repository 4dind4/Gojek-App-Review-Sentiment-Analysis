# 🤖 Gojek App Review Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=for-the-badge&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.6-green?style=for-the-badge&logo=scikit-learn)
![Keras](https://img.shields.io/badge/Keras-3.13-red?style=for-the-badge&logo=keras)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?style=for-the-badge&logo=googlecolab)

**Proyek Klasifikasi Sentimen Ulasan Aplikasi Gojek dari Google Play Store**
*Dicoding — Belajar Pengembangan Machine Learning*

</div>

---

## 📋 Daftar Isi

- [Situasi (Situation)](#-situasi-situation)
- [Tugas (Task)](#-tugas-task)
- [Pendekatan (Action)](#-pendekatan-action)
- [Hasil (Result)](#-hasil-result)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dependensi](#-dependensi)

---

## 🌍 Situasi *(Situation)*

Gojek merupakan salah satu aplikasi super-app terbesar di Asia Tenggara dengan jutaan pengguna aktif setiap harinya. Setiap harinya, ribuan ulasan dari pengguna mengalir masuk ke Google Play Store, mengandung informasi berharga tentang kepuasan, keluhan, dan harapan pengguna terhadap layanan Gojek.

Namun, **volume data yang sangat besar** membuat proses analisis manual menjadi tidak efisien dan tidak skalabel. Diperlukan sebuah sistem otomatis yang mampu:

- Mengumpulkan ulasan secara otomatis
- Mengklasifikasikan sentimen secara akurat
- Menangani karakteristik khusus teks berbahasa Indonesia informal

### Karakteristik Data yang Menantang

| Karakteristik | Deskripsi |
|---|---|
| 📝 *Short Text* | Rata-rata panjang ulasan hanya ~10 kata |
| ⚖️ *Imbalanced Data* | Distribusi: Positif (59.7%) > Negatif (35.7%) > Netral (4.6%) |
| 🗣️ Bahasa Informal | Banyak singkatan, typo, dan slang khas pengguna Indonesia |

---

## 📌 Tugas *(Task)*

Membangun sistem analisis sentimen end-to-end yang mampu:

1. **Mengumpulkan data** ulasan Gojek dari Google Play Store secara otomatis
2. **Melakukan preprocessing** teks berbahasa Indonesia informal
3. **Melatih dan membandingkan** beberapa model Machine Learning dan Deep Learning
4. **Mengklasifikasikan sentimen** ulasan ke dalam 3 kelas:
   - 🟢 **Positif** — Ulasan dengan rating 4–5 bintang
   - 🟡 **Netral** — Ulasan dengan rating 3 bintang
   - 🔴 **Negatif** — Ulasan dengan rating 1–2 bintang

---

## 🛠️ Pendekatan *(Action)*

### 1. Pengumpulan Data (`Scraping_Data.ipynb`)

Data dikumpulkan menggunakan library `google-play-scraper` langsung dari Google Play Store.

```python
from google_play_scraper import reviews, Sort

result, _ = reviews(
    'com.gojek.app',   # App ID Gojek di Play Store
    lang='id',          # Bahasa Indonesia
    country='id',
    sort=Sort.NEWEST,
    count=6000          # Total 6.000 ulasan terbaru
)
```

**Hasil scraping:** 6.000 ulasan dengan 11 kolom fitur (reviewId, userName, content, score, dll.)

---

### 2. Eksplorasi Data (EDA)

- Distribusi rating dianalisis menggunakan `seaborn` countplot
- Labeling sentimen otomatis berdasarkan rating:

```python
df['sentiment'] = df['score'].apply(lambda x:
    'negative' if x <= 2 else
    'neutral'  if x == 3 else
    'positive'
)
```

| Sentimen | Jumlah | Persentase |
|---|---|---|
| Positif | 3.581 | 59.7% |
| Negatif | 2.142 | 35.7% |
| Netral | 277 | 4.6% |

---

### 3. Preprocessing Teks

Pipeline preprocessing yang diterapkan:

| Langkah | Deskripsi |
|---|---|
| Lowercasing | Mengubah teks menjadi huruf kecil |
| Hapus URL & Mention | Membersihkan link dan @mention |
| Hapus Angka & Simbol | Menghilangkan karakter non-alfabet |
| Hapus Stopword | Menggunakan daftar stopword Bahasa Indonesia |
| Stemming | Menggunakan **PySastrawi** untuk stemming Bahasa Indonesia |

---

### 4. Pemodelan

Proyek ini mengimplementasikan dan membandingkan **3 model**:

#### Model 1 — Naive Bayes + TF-IDF
```
Vectorizer  : TfidfVectorizer
Classifier  : MultinomialNB
Imbalance   : SMOTE (oversampling)
```

#### Model 2 — Logistic Regression + TF-IDF
```
Vectorizer  : TfidfVectorizer
Classifier  : LogisticRegression (class_weight='balanced')
Solver      : lbfgs, max_iter=1000
```

#### Model 3 — LSTM (Deep Learning)
```
Architecture : Embedding -> LSTM(64) -> Dense(3, softmax)
Optimizer    : Adam
Loss         : sparse_categorical_crossentropy
Epochs       : 10
Batch Size   : 32
```

---

## 📊 Hasil *(Result)*

### Performa Model LSTM (Model Terbaik)

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 96.56% | **97.29%** |
| 2 | 97.73% | 96.25% |
| 3 | 97.94% | 95.83% |
| 5 | 98.78% | 95.31% |
| 10 | 99.17% | 94.48% |

> ✅ Model mencapai **val_accuracy tertinggi ~97.29%** pada epoch pertama, menunjukkan model yang kuat dalam mengklasifikasikan sentimen ulasan Gojek berbahasa Indonesia.

### Alasan Pemilihan Metode

| Faktor | Penjelasan |
|---|---|
| TF-IDF | Efektif untuk *short text classification* |
| Logistic Regression | Kuat untuk *sparse features* hasil TF-IDF |
| SMOTE | Mengatasi ketidakseimbangan kelas (*class imbalance*) |
| PySastrawi | Stemming akurat untuk Bahasa Indonesia |
| LSTM | Menangkap konteks sekuensial dalam teks |

---

## 📁 Struktur Proyek

```
Proyek Analisis Sentimen/
|
├── Scraping_Data.ipynb    # Notebook untuk scraping data dari Play Store
├── Build_Model.ipynb      # Notebook utama: EDA, preprocessing, dan pemodelan
├── gojek_reviews.csv      # Dataset hasil scraping (6.000 ulasan)
└── requirements.txt       # Daftar dependensi Python
```

---

## 🚀 Cara Penggunaan

### Prasyarat

Proyek ini dirancang untuk dijalankan di **Google Colab** (direkomendasikan) atau lingkungan Python lokal.

---

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/4dind4/Gojek-App-Review-Sentiment-Analysis.git
cd Gojek-App-Review-Sentiment-Analysis
```

---

### Langkah 2 — Buka di Google Colab (Direkomendasikan)

1. Buka [Google Colab](https://colab.research.google.com/)
2. Pilih **File → Upload Notebook**
3. Upload file `Build_Model.ipynb`
4. Upload dataset `gojek_reviews.csv` ke Google Drive kamu
5. Mount Google Drive di awal notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

6. Sesuaikan path dataset di cell kedua:

```python
df = pd.read_csv("/content/drive/MyDrive/[FOLDER_KAMU]/gojek_reviews.csv")
```

---

### Langkah 3 — Jalankan Notebook

#### Jika ingin melakukan scraping data baru:

Buka `Scraping_Data.ipynb` terlebih dahulu dan jalankan semua cell secara berurutan:

```
Cell 1 → Install dependencies (google-play-scraper, PySastrawi)
Cell 2 → Import library
Cell 3 → Scraping 6.000 ulasan terbaru dari Play Store
Cell 4 → Simpan ke gojek_reviews.csv
```

#### Jika ingin langsung membangun model (gunakan dataset yang tersedia):

Buka `Build_Model.ipynb` dan jalankan semua cell:

```
1. Mount Google Drive & Load Dataset
2. Eksplorasi Data (EDA) — distribusi rating, statistik deskriptif
3. Preprocessing Teks — cleaning, stopword removal, stemming
4. Pembagian Data — Train/Test Split (80/20)
5. Model 1: Naive Bayes + TF-IDF + SMOTE
6. Model 2: Logistic Regression + TF-IDF
7. Model 3: LSTM (Deep Learning)
8. Evaluasi & Visualisasi — accuracy, classification report, training curve
```

---

### Langkah 4 — Menggunakan Model untuk Prediksi

Setelah model dilatih, kamu bisa melakukan prediksi sentimen pada teks baru:

#### Prediksi dengan Logistic Regression / Naive Bayes:

```python
teks_baru = ["gojek sekarang makin bagus dan cepat!"]
teks_preprocessed = [preprocess(t) for t in teks_baru]
teks_vectorized = tfidf.transform(teks_preprocessed)
prediksi = model_lr.predict(teks_vectorized)
print("Sentimen:", prediksi[0])
# Output: positive
```

#### Prediksi dengan LSTM:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

teks_baru = ["aplikasi sering error dan lambat"]
teks_seq = tokenizer.texts_to_sequences(teks_baru)
teks_padded = pad_sequences(teks_seq, maxlen=MAX_LEN)
prediksi = model_lstm.predict(teks_padded)
label = label_encoder.inverse_transform([prediksi.argmax()])
print("Sentimen:", label[0])
# Output: negative
```

---

## 📦 Dependensi

Install dependensi utama yang dibutuhkan:

```bash
pip install google-play-scraper PySastrawi
pip install pandas numpy matplotlib seaborn scikit-learn
pip install imbalanced-learn tensorflow keras
```

### Library Utama

| Library | Versi | Kegunaan |
|---|---|---|
| `google-play-scraper` | 1.2.7 | Scraping ulasan Play Store |
| `PySastrawi` | 1.2.0 | Stemming Bahasa Indonesia |
| `pandas` | 2.2.2 | Manipulasi data |
| `numpy` | 2.0.2 | Komputasi numerik |
| `scikit-learn` | 1.6.1 | Model ML (NB, LR, TF-IDF) |
| `imbalanced-learn` | 0.14.1 | SMOTE untuk imbalanced data |
| `tensorflow` | 2.20.0 | Deep Learning (LSTM) |
| `keras` | 3.13.2 | API Deep Learning |
| `matplotlib` | 3.10.0 | Visualisasi data |
| `seaborn` | 0.13.2 | Visualisasi statistik |

---

## 📝 Catatan Penting

> **Path Dataset**: Pastikan untuk menyesuaikan path file `gojek_reviews.csv` sesuai lokasi penyimpanan di Google Drive kamu pada cell pertama `Build_Model.ipynb`.

> **Google Colab GPU**: Proyek ini dioptimalkan untuk Google Colab. Aktifkan GPU untuk pelatihan LSTM yang lebih cepat di **Runtime → Change runtime type → GPU**.

> **Scraping Data Terbaru**: Untuk mendapatkan data ulasan terbaru, jalankan `Scraping_Data.ipynb` dan update path dataset di `Build_Model.ipynb`.

---

## 👤 Author

**Dicoding — Proyek Akhir Belajar Pengembangan Machine Learning**

---

<div align="center">
  <i>Dibuat dengan ❤️ untuk keperluan pembelajaran Machine Learning</i>
</div>
