import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from scipy.sparse import issparse

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
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing pada data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data yang akan diproses
    test_size : float, default=0.2
        Ukuran data test relatif terhadap keseluruhan dataset
    random_state : int, default=42
        Random seed untuk reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Data yang sudah di-split untuk training dan testing
    """
    try:
        logger.info("Memulai preprocessing data...")
        
        # Cek jika data kosong
        if df.empty:
            raise ValueError("DataFrame kosong")
        
        # Identifikasi kolom target (y atau target)
        target_col = 'y'
        if target_col not in df.columns and 'target' in df.columns:
            target_col = 'target'
            logger.info(f"Menggunakan kolom '{target_col}' sebagai target")
        
        if target_col not in df.columns:
            raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset. Kolom yang tersedia: {df.columns.tolist()}")
        
        # Pisahkan fitur dan target
        X = df.drop(target_col, axis=1)
        
        # Konversi target ke numerik jika perlu
        y = df[target_col]
        if target_col == 'y' and y.dtype == 'object':
            # Jika target adalah 'y' dan berisi string, konversi yes/no ke 1/0
            y = y.map({'yes': 1, 'no': 0})
            logger.info("Mengonversi nilai 'yes'/'no' pada kolom target ke 1/0")
        
        logger.info(f"Jumlah fitur: {X.shape[1]}")
        logger.info(f"Distribusi target: {y.value_counts().to_dict()}")
        
        # Identifikasi kolom numerik dan kategorikal
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Kolom numerik: {numerical_cols}")
        logger.info(f"Kolom kategorikal: {categorical_cols}")
        
        # Buat preprocessor menggunakan ColumnTransformer
        transformers = []
        
        if numerical_cols:
            transformers.append(('num', StandardScaler(), numerical_cols))
            
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'  # Sisa kolom akan dilewatkan apa adanya
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit dan transform data
        logger.info("Menerapkan transformasi pada data...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Simpan preprocessor untuk digunakan nanti
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        
        # Simpan scaler untuk kompatibilitas backward
        joblib.dump(preprocessor, 'models/scaler.pkl')
        
        # Log informasi
        logger.info(f"Ukuran X_train: {X_train_processed.shape}")
        logger.info(f"Ukuran X_test: {X_test_processed.shape}")
        logger.info("Preprocessing data selesai")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error dalam preprocessing data: {str(e)}")
        raise

# Fungsi untuk melatih model
def train_model(X_train, y_train, model_type='random_forest'):
    """
    Melatih model ML menggunakan data training
    """
    try:
        logger.info(f"Melatih model {model_type}...")
        
        # Konversi X_train ke array jika dalam format sparse matrix
        if issparse(X_train):
            logger.info("Input data dalam format sparse matrix")
        
        # Pilih model berdasarkan tipe
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            logger.warning(f"Tipe model '{model_type}' tidak dikenal. Menggunakan RandomForest sebagai default.")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Latih model
        logger.info("Mulai pelatihan model...")
        model.fit(X_train, y_train)
        logger.info("Pelatihan model selesai")
        
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
        
        # Konversi X_test ke array jika dalam format sparse matrix
        if issparse(X_test):
            logger.info("Data test dalam format sparse matrix")
        
        # Prediksi
        y_pred = model.predict(X_test)
        
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
            
            # Feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importance untuk 20 fitur teratas (atau semua jika < 20)
            n_features = min(20, len(importances))
            plt.bar(range(n_features), importances[indices[:n_features]])
            plt.title('Feature Importances (Top 20)')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
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
def run_pipeline(data_path='data/test_data.csv', model_type='random_forest'):
    """
    Menjalankan seluruh ML pipeline, dari load data hingga evaluasi model
    """
    try:
        logger.info("Memulai pipeline ML...")
        
        # Buat direktori yang diperlukan
        for dir_name in ['data', 'models', 'reports']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        
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