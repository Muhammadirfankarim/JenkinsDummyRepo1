# Pipeline Machine Learning dengan Jenkins

Repositori ini berisi kode untuk pipeline machine learning otomatis menggunakan Jenkins di lingkungan Windows.

## Komponen Utama

1. **ml_pipeline.py**: Skrip utama untuk pipeline machine learning, termasuk:
   - Load data
   - Preprocessing
   - Training model
   - Evaluasi model

2. **model_monitoring.py**: Skrip untuk memantau performa model dan pergeseran data.

3. **test_pipeline.py**: Tes unit untuk validasi pipeline ML.

4. **Jenkinsfile**: Konfigurasi Jenkins pipeline untuk CI/CD.

## Struktur Direktori

```
├── data/                # Dataset
├── models/              # Model terlatih dan preprocessor
├── reports/             # Laporan evaluasi dan monitoring
├── ml_pipeline.py       # Skrip pipeline ML
├── model_monitoring.py  # Skrip monitoring model
├── test_pipeline.py     # Tes unit
├── Jenkinsfile          # Konfigurasi Jenkins
└── requirements.txt     # Dependensi Python
```

## Cara Penggunaan

### Penyiapan Lingkungan

1. Pastikan Docker diinstal dan berjalan di mesin Jenkins
2. Konfigurasi Jenkins untuk dapat menjalankan Docker
3. Buat pipeline baru di Jenkins menggunakan Jenkinsfile

### Menjalankan Pipeline

Pipeline Jenkins terdiri dari beberapa tahap:
1. **Setup Environment**: Menyiapkan lingkungan dan direktori yang diperlukan
2. **Run Tests**: Menjalankan tes unit untuk memvalidasi pipeline
3. **Train Model**: Melatih model machine learning
4. **Model Monitoring**: Memantau performa model dan pergeseran data

## Dependensi

Lihat `requirements.txt` untuk daftar lengkap dependensi Python.

## Perubahan Terbaru

### Penggunaan Docker untuk Kompatibilitas Windows

Pipeline ini menggunakan Docker untuk mengisolasi lingkungan Python, yang memungkinkan:
1. Kompatibilitas lintas platform (Windows/Linux)
2. Konsistensi lingkungan
3. Pengelolaan dependensi yang lebih mudah

### Penanganan Fitur Kategorikal

Pipeline telah diperbarui untuk menangani fitur kategorikal dengan benar menggunakan:
1. `OneHotEncoder` untuk fitur kategorikal
2. `StandardScaler` untuk fitur numerik
3. `ColumnTransformer` untuk menggabungkan transformasi

### Penyesuaian untuk Windows

Beberapa penyesuaian telah dilakukan untuk memastikan kompatibilitas dengan Jenkins di Windows:
1. Penggunaan `%CD%` atau path absolut untuk mounting volume Docker
2. Penggunaan perintah `bat` untuk eksekusi shell Windows
3. Alternatif path absolut tersedia sebagai komentar jika path relatif tidak berfungsi

## Resolusi Masalah

Jika mengalami masalah dengan path Docker di Windows, coba gunakan path absolut dengan menghapus komentar pada alternatif di `Jenkinsfile`. 