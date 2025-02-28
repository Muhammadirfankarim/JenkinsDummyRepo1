import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fungsi untuk memuat data
def load_data(filepath='data/bank.csv'):
    """
    Memuat dataset dari filepath yang diberikan
    """
    try:
        logger.info(f"Memuat data dari {filepath}")
        # Buat direktori jika tidak ada
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Jika file tidak ada, buat dataset dummy untuk keperluan demo
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} tidak ditemukan. Membuat dataset dummy...")
            # Membuat dataset dummy
            np.random.seed(42)
            n_samples = 1000
            X = np.random.randn(n_samples, 10)
            y = (X[:, 0] + X[:, 1]**2 + np.random.randn(n_samples) * 0.5) > 0
            y = y.astype(int)
            
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
            df['target'] = y
            
            # Simpan dataset
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset dummy disimpan ke {filepath}")
            return df
        
        # Coba baca file dengan pemisah berbeda
        try:
            # Pertama coba dengan pemisah koma (default)
            df = pd.read_csv(filepath)
            
            # Periksa apakah hanya ada satu kolom dan berisi titik koma
            if len(df.columns) == 1 and any(';' in str(col) for col in df.columns):
                logger.info("Mendeteksi format CSV dengan pemisah titik koma, membaca ulang file...")
                df = pd.read_csv(filepath, sep=';')
            
            # Periksa apakah kolom 'target' ada
            if 'target' not in df.columns and 'y' in df.columns:
                logger.info("Mengubah nama kolom 'y' menjadi 'target'")
                df = df.rename(columns={'y': 'target'})
                
            # Jika masih tidak ada kolom target, buat dataset dummy
            if 'target' not in df.columns:
                logger.warning("Kolom 'target' tidak ditemukan di dataset. Membuat dataset dummy...")
                # Membuat dataset dummy
                np.random.seed(42)
                n_samples = 1000
                X = np.random.randn(n_samples, 10)
                y = (X[:, 0] + X[:, 1]**2 + np.random.randn(n_samples) * 0.5) > 0
                y = y.astype(int)
                
                df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
                df['target'] = y
                
                # Simpan dataset
                df.to_csv(filepath, index=False)
                logger.info(f"Dataset dummy disimpan ke {filepath}")
        except Exception as e:
            logger.error(f"Error membaca file CSV: {str(e)}. Membuat dataset dummy...")
            # Membuat dataset dummy
            np.random.seed(42)
            n_samples = 1000
            X = np.random.randn(n_samples, 10)
            y = (X[:, 0] + X[:, 1]**2 + np.random.randn(n_samples) * 0.5) > 0
            y = y.astype(int)
            
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
            df['target'] = y
            
            # Simpan dataset
            df.to_csv(filepath, index=False)
            logger.info(f"Dataset dummy disimpan ke {filepath}")
        
        logger.info(f"Data berhasil dimuat, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error memuat data: {str(e)}")
        raise

# Fungsi untuk preprocessing data
def preprocess_data(df):
    """
    Melakukan preprocessing pada data dan membagi menjadi train dan test set
    """
    try:
        logger.info("Melakukan preprocessing data...")
        
        # Verifikasi kolom target
        target_col = 'target'
        if target_col not in df.columns and 'y' in df.columns:
            target_col = 'y'
            logger.info("Menggunakan kolom 'y' sebagai target")
        
        if target_col not in df.columns:
            raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset")
        
        # Memisahkan features dan target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Standarisasi fitur (skip jika semua kolom adalah kategorikal)
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
            
            # Simpan scaler
            if not os.path.exists('models'):
                os.makedirs('models')
            joblib.dump(scaler, 'models/scaler.pkl')
            logger.info("Scaler disimpan ke models/scaler.pkl")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Data dibagi menjadi: X_train {X_train.shape}, X_test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

# Fungsi untuk melatih model
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Melatih model ML menggunakan data training
    """
    try:
        logger.info(f"Melatih model {model_type}...")
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42)
        else:
            logger.warning(f"Tipe model '{model_type}' tidak dikenal. Menggunakan RandomForest sebagai default.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Standarisasi data jika belum dilakukan
        if isinstance(X_train, pd.DataFrame):
            # Fitur sudah distandarisasi di preprocess_data
            X_train_data = X_train
        else:
            # Jika X_train bukan DataFrame, gunakan apa adanya
            X_train_data = X_train
        
        # Latih model
        model.fit(X_train_data, y_train)
        
        # Simpan model
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(model, 'models/model.pkl')
        logger.info("Model disimpan ke models/model.pkl")
        
        return model
    except Exception as e:
        logger.error(f"Error melatih model: {str(e)}")
        raise

# Fungsi untuk evaluasi model
def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model dan mengembalikan metrik performa
    """
    try:
        logger.info("Evaluasi model...")
        
        # Standarisasi data jika belum dilakukan
        if isinstance(X_test, pd.DataFrame):
            # Fitur sudah distandarisasi di preprocess_data
            X_test_data = X_test
        else:
            # Jika X_test bukan DataFrame, gunakan apa adanya
            X_test_data = X_test
        
        # Prediksi
        y_pred = model.predict(X_test_data)
        
        # Evaluasi metrik
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Buat direktori reports jika belum ada
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Plot feature importance (jika model mendukung)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            if isinstance(X_test, pd.DataFrame):
                features = X_test.columns
            else:
                features = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            importances = pd.Series(model.feature_importances_, index=features)
            importances = importances.sort_values(ascending=False)
            importances.plot(kind='bar')
            plt.title('Feature Importances')
            plt.tight_layout()
            plt.savefig('reports/feature_importance.png')
            logger.info("Feature importance plot disimpan ke reports/feature_importance.png")
        
        # Simpan metrik ke file CSV
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        metrics_df = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
            'value': [accuracy, precision, recall, f1]
        })
        metrics_df.to_csv(f'reports/metrics_{timestamp}.csv', index=False)
        logger.info(f"Metrik disimpan ke reports/metrics_{timestamp}.csv")
        
        # Return metrik dalam bentuk dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error evaluasi model: {str(e)}")
        raise

# Fungsi utama untuk menjalankan pipeline
def run_pipeline(data_path='data/bank.csv', model_type='random_forest'):
    """
    Menjalankan seluruh ML pipeline, dari load data hingga evaluasi model
    """
    try:
        logger.info("Memulai pipeline ML...")
        
        # 1. Load Data
        logger.info("1. Memuat data...")
        df = load_data(data_path)
        
        # 2. Preprocessing Data
        logger.info("2. Melakukan preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # 3. Train Model
        logger.info("3. Melatih model...")
        model = train_model(X_train, y_train, model_type)
        
        # 4. Evaluasi Model
        logger.info("4. Mengevaluasi model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        logger.info("Pipeline ML selesai!")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error dalam pipeline: {str(e)}")
        # Kembalikan dictionary kosong jika terjadi error
        return {
            'error': str(e),
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }

# Main entry point
if __name__ == "__main__":
    run_pipeline() 