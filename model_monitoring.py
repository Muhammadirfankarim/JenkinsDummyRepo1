import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import joblib
from datetime import datetime
import logging
from scipy.sparse import issparse

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_monitoring')

def load_latest_metrics():
    """
    Memuat metrics dari file CSV terbaru di direktori reports
    """
    try:
        logger.info("Mencari file metrics terbaru")
        metrics_files = glob.glob('reports/metrics_*.csv')
        
        if not metrics_files:
            logger.warning("Tidak ada file metrics yang ditemukan")
            return None
        
        # Cari file terbaru berdasarkan tanggal di nama file
        latest_file = max(metrics_files, key=os.path.getctime)
        logger.info(f"File metrics terbaru: {latest_file}")
        
        # Load metrics
        metrics_df = pd.read_csv(latest_file)
        return metrics_df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error memuat metrics: {str(e)}")
        return None

def load_all_metrics():
    """
    Memuat semua metrics dari file CSV di direktori reports
    """
    try:
        logger.info("Memuat semua file metrics")
        metrics_files = glob.glob('reports/metrics_*.csv')
        
        if not metrics_files:
            logger.warning("Tidak ada file metrics yang ditemukan")
            return None
        
        # Urutkan file berdasarkan waktu pembuatan
        metrics_files.sort(key=os.path.getctime)
        
        # Ekstrak timestamps dari nama file
        timestamps = []
        metrics_dfs = []
        
        for file in metrics_files:
            # Ekstrak timestamp dari nama file (format: metrics_YYYYMMDD_HHMMSS.csv)
            ts_str = file.split('metrics_')[1].split('.csv')[0]
            timestamp = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
            timestamps.append(timestamp)
            
            # Load metrics
            df = pd.read_csv(file)
            metrics_dfs.append(df)
        
        # Gabungkan metrics dengan timestamps
        all_metrics = pd.concat(metrics_dfs, ignore_index=True)
        all_metrics['timestamp'] = timestamps
        
        return all_metrics
    except Exception as e:
        logger.error(f"Error memuat semua metrics: {str(e)}")
        return None

def generate_performance_report():
    """
    Membuat laporan performa model berdasarkan metrics yang telah dikumpulkan
    """
    try:
        logger.info("Membuat laporan performa model")
        
        # Memuat semua metrics
        all_metrics = load_all_metrics()
        
        if all_metrics is None or len(all_metrics) == 0:
            logger.warning("Tidak ada metrics yang tersedia untuk laporan")
            return False
        
        # Buat direktori untuk laporan jika belum ada
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Buat plot metrics dari waktu ke waktu
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            ax.plot(all_metrics['timestamp'], all_metrics[metric], marker='o')
            ax.set_title(f'{metric.capitalize()} over Time')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
        
        fig.tight_layout()
        fig.savefig('reports/performance_trends.png')
        logger.info("Laporan performa disimpan ke reports/performance_trends.png")
        
        # Simpan laporan rangkuman ke CSV
        summary = all_metrics.describe()
        summary.to_csv('reports/metrics_summary.csv')
        logger.info("Rangkuman metrics disimpan ke reports/metrics_summary.csv")
        
        # Tampilkan metrics terbaru
        latest_metrics = all_metrics.iloc[-1]
        logger.info("Metrics terbaru:")
        for metric in metrics:
            logger.info(f"{metric.capitalize()}: {latest_metrics[metric]:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error membuat laporan performa: {str(e)}")
        return False

def check_model_drift(threshold=0.05):
    """
    Memeriksa drift model dengan membandingkan metrics terbaru dengan sebelumnya
    Mengembalikan True jika drift terdeteksi (penurunan performa > threshold)
    """
    try:
        logger.info(f"Memeriksa model drift dengan threshold {threshold}")
        
        # Memuat semua metrics
        all_metrics = load_all_metrics()
        
        if all_metrics is None or len(all_metrics) < 2:
            logger.warning("Tidak cukup data untuk mendeteksi drift")
            return False
        
        # Ambil metrics dua run terakhir
        last_metrics = all_metrics.iloc[-1]
        prev_metrics = all_metrics.iloc[-2]
        
        # Periksa penurunan pada tiap metrik
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        drift_detected = False
        
        for metric in metrics:
            delta = last_metrics[metric] - prev_metrics[metric]
            if delta < -threshold:  # Jika penurunan melebihi threshold
                logger.warning(f"Drift terdeteksi pada {metric}: {delta:.4f} (penurunan > {threshold})")
                drift_detected = True
        
        if not drift_detected:
            logger.info("Tidak ada drift yang terdeteksi")
        
        return drift_detected
    except Exception as e:
        logger.error(f"Error memeriksa model drift: {str(e)}")
        return False

def generate_alert(message):
    """
    Membuat file alert dengan pesan tertentu
    """
    try:
        logger.warning(f"Membuat alert: {message}")
        with open(f'reports/ALERT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write(f"ALERT: {message}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Tambahkan metrics terbaru jika tersedia
            latest_metrics = load_latest_metrics()
            if latest_metrics:
                f.write("\nLatest Metrics:\n")
                for key, value in latest_metrics.items():
                    if key != 'timestamp':
                        f.write(f"{key}: {value:.4f}\n")
        
        logger.info("File alert berhasil dibuat")
        return True
    except Exception as e:
        logger.error(f"Error membuat alert: {str(e)}")
        return False

def run_monitoring():
    """
    Menjalankan pemantauan model
    """
    try:
        logger.info("Memulai pemantauan model")
        
        # Membuat laporan performa
        generate_performance_report()
        
        # Memeriksa model drift
        drift_detected = check_model_drift()
        
        if drift_detected:
            generate_alert("Model drift terdeteksi! Performa model menurun secara signifikan")
        
        logger.info("Pemantauan model selesai")
        return True
    except Exception as e:
        logger.error(f"Error dalam pemantauan model: {str(e)}")
        return False

def monitor_data_drift(new_data_path, reference_data_path='data/reference_data.csv'):
    """
    Memantau pergeseran data antara dataset referensi dan dataset baru
    
    Parameters:
    -----------
    new_data_path : str
        Path ke dataset baru
    reference_data_path : str
        Path ke dataset referensi
        
    Returns:
    --------
    drift_metrics : dict
        Metrik-metrik pergeseran data
    """
    try:
        logger.info("Memulai monitoring pergeseran data...")
        
        # Periksa apakah file referensi ada
        if not os.path.exists(reference_data_path):
            # Jika tidak ada, gunakan data baru sebagai referensi
            logger.warning(f"File referensi {reference_data_path} tidak ditemukan. Menggunakan data baru sebagai referensi.")
            if not os.path.exists(os.path.dirname(reference_data_path)):
                os.makedirs(os.path.dirname(reference_data_path))
            
            new_data = pd.read_csv(new_data_path)
            new_data.to_csv(reference_data_path, index=False)
            logger.info(f"Data baru disimpan sebagai referensi di {reference_data_path}")
            return {'message': 'Data referensi dibuat'}
        
        # Baca data
        reference_data = pd.read_csv(reference_data_path)
        new_data = pd.read_csv(new_data_path)
        
        # Pastikan kolom sama
        if set(reference_data.columns) != set(new_data.columns):
            logger.error("Struktur kolom antara data referensi dan data baru berbeda")
            return {'error': 'Struktur kolom berbeda'}
        
        drift_metrics = {}
        
        # Hitung statistik dasar untuk kolom numerik
        for col in reference_data.select_dtypes(include=['int64', 'float64']).columns:
            ref_mean = reference_data[col].mean()
            new_mean = new_data[col].mean()
            mean_diff_pct = ((new_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
            
            ref_std = reference_data[col].std()
            new_std = new_data[col].std()
            std_diff_pct = ((new_std - ref_std) / ref_std) * 100 if ref_std != 0 else 0
            
            drift_metrics[col] = {
                'mean_diff_pct': mean_diff_pct,
                'std_diff_pct': std_diff_pct
            }
            
            # Log perubahan signifikan
            if abs(mean_diff_pct) > 10:
                logger.warning(f"Perubahan signifikan pada mean {col}: {mean_diff_pct:.2f}%")
            if abs(std_diff_pct) > 20:
                logger.warning(f"Perubahan signifikan pada std {col}: {std_diff_pct:.2f}%")
        
        # Identifikasi kolom target
        target_col = 'y'
        if target_col not in reference_data.columns and 'target' in reference_data.columns:
            target_col = 'target'
            
        # Hitung distribusi untuk kolom kategorikal
        for col in reference_data.select_dtypes(include=['object', 'category']).columns:
            if col == target_col:  # Lewati kolom target
                continue
                
            ref_dist = reference_data[col].value_counts(normalize=True).to_dict()
            new_dist = new_data[col].value_counts(normalize=True).to_dict()
            
            # Hitung distribusi kategorikal
            drift_metrics[col] = {'category_drift': {}}
            
            for category in set(list(ref_dist.keys()) + list(new_dist.keys())):
                ref_pct = ref_dist.get(category, 0) * 100
                new_pct = new_dist.get(category, 0) * 100
                diff_pct = new_pct - ref_pct
                
                drift_metrics[col]['category_drift'][category] = {
                    'reference_pct': ref_pct,
                    'new_pct': new_pct,
                    'diff_pct': diff_pct
                }
                
                # Log perubahan signifikan
                if abs(diff_pct) > 5:
                    logger.warning(f"Perubahan signifikan pada {col}={category}: {diff_pct:.2f}%")
        
        # Buat report
        if not os.path.exists('reports'):
            os.makedirs('reports')
            
        report_file = f'reports/drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        pd.DataFrame(drift_metrics).to_json(report_file)
        logger.info(f"Report pergeseran data disimpan ke {report_file}")
        
        return drift_metrics
        
    except Exception as e:
        logger.error(f"Error dalam monitoring pergeseran data: {str(e)}")
        return {'error': str(e)}

def monitor_model_performance(data_path, model_path='models/model.pkl', preprocessor_path='models/preprocessor.pkl'):
    """
    Memantau performa model pada data baru
    
    Parameters:
    -----------
    data_path : str
        Path ke dataset baru untuk evaluasi
    model_path : str
        Path ke model tersimpan
    preprocessor_path : str
        Path ke preprocessor tersimpan
    
    Returns:
    --------
    performance_metrics : dict
        Metrik performa model
    """
    try:
        logger.info("Memulai monitoring performa model...")
        
        # Periksa keberadaan model dan preprocessor
        if not os.path.exists(model_path):
            logger.error(f"Model tidak ditemukan di {model_path}")
            return {'error': 'Model tidak ditemukan'}
            
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor tidak ditemukan di {preprocessor_path}")
            return {'error': 'Preprocessor tidak ditemukan'}
        
        # Load model dan preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Identifikasi kolom target
        target_col = 'y'
        if target_col not in data.columns and 'target' in data.columns:
            target_col = 'target'
            logger.info(f"Menggunakan kolom '{target_col}' sebagai target")
        
        if target_col not in data.columns:
            logger.error(f"Kolom target '{target_col}' tidak ditemukan")
            return {'error': 'Kolom target tidak ditemukan'}
            
        # Pisahkan fitur dan target
        X = data.drop(target_col, axis=1)
        
        # Konversi target ke numerik jika perlu
        y_true = data[target_col]
        if target_col == 'y' and y_true.dtype == 'object':
            # Jika target adalah 'y' dan berisi string, konversi yes/no ke 1/0
            y_true = y_true.map({'yes': 1, 'no': 0})
            logger.info("Mengonversi nilai 'yes'/'no' pada kolom target ke 1/0")
        
        # Preprocessing data
        X_processed = preprocessor.transform(X)
        
        # Periksa apakah output adalah sparse matrix
        if issparse(X_processed):
            logger.info("Data yang diproses adalah sparse matrix")
        
        # Prediksi
        y_pred = model.predict(X_processed)
        
        # Hitung metrik
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        performance_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Log metrik
        for metric, value in performance_metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Buat report
        if not os.path.exists('reports'):
            os.makedirs('reports')
            
        report_file = f'reports/performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        pd.DataFrame([performance_metrics]).to_json(report_file, orient='records')
        logger.info(f"Report performa model disimpan ke {report_file}")
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Error dalam monitoring performa model: {str(e)}")
        return {'error': str(e)}

def main():
    """
    Fungsi utama untuk menjalankan monitoring
    """
    try:
        # Set parameter
        data_path = 'data/test_data.csv'  # Path ke data yang akan dimonitor
        
        # Jalankan monitoring data drift
        logger.info("Menjalankan monitoring pergeseran data...")
        drift_metrics = monitor_data_drift(data_path)
        
        # Jalankan monitoring performa model
        logger.info("Menjalankan monitoring performa model...")
        performance_metrics = monitor_model_performance(data_path)
        
        logger.info("Monitoring selesai")
        
    except Exception as e:
        logger.error(f"Error dalam monitoring: {str(e)}")

if __name__ == "__main__":
    main() 