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
import time

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
def load_data(data_path):
    """
    Fungsi untuk memuat data dari file CSV.
    Jika file tidak ditemukan, akan dibuat data dummy.
    
    Args:
        data_path (str): Path ke file CSV
        
    Returns:
        pandas.DataFrame: DataFrame yang berisi data
    """
    try:
        logging.info(f"Loading data from {data_path}")
        
        # Coba baca file
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully with shape {df.shape}")
            
            # Jika menggunakan format bank.csv, coba ubah kolom 'y' menjadi 'target'
            if 'y' in df.columns and 'target' not in df.columns:
                df = df.rename(columns={'y': 'target'})
                logging.info("Renamed column 'y' to 'target'")
            
            # Pastikan ada kolom target
            if 'target' not in df.columns:
                logging.warning(f"Target column 'target' not found in {list(df.columns)}")
                # Gunakan kolom terakhir sebagai target
                if len(df.columns) > 1:
                    df = df.rename(columns={df.columns[-1]: 'target'})
                    logging.info(f"Renamed column '{df.columns[-1]}' to 'target'")
                else:
                    # Jika hanya ada satu kolom, buat kolom target
                    df['target'] = np.random.randint(0, 2, size=len(df))
                    logging.info("Added random target column")
            
            return df
        else:
            # Jika file tidak ditemukan, gunakan test_data.csv jika ada
            if os.path.exists('data/test_data.csv'):
                logging.info(f"File {data_path} not found, using data/test_data.csv instead")
                df = pd.read_csv('data/test_data.csv')
                return df
                
            # Jika tidak, buat data dummy
            logging.warning(f"File {data_path} not found, creating dummy data")
            # Buat data dummy
            X = np.random.rand(100, 2)
            y = np.random.randint(0, 2, size=100)
            
            df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
            df['target'] = y
            
            # Simpan data dummy
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            try:
                df.to_csv(data_path, index=False)
                logging.info(f"Dummy data saved to {data_path}")
            except Exception as e:
                logging.error(f"Error saving dummy data: {e}")
            
            return df
    except Exception as e:
        logging.error(f"Error in load_data: {e}")
        # Buat data dummy dalam kasus error
        X = np.random.rand(100, 2)
        y = np.random.randint(0, 2, size=100)
        
        df = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        df['target'] = y
        
        return df

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
        logging.info("Preprocessing data...")
        
        # Periksa apakah ada kolom target
        if 'target' not in df.columns:
            if 'y' in df.columns:
                # Rename kolom jika menggunakan format bank.csv
                df = df.rename(columns={'y': 'target'})
                logging.info("Renamed column 'y' to 'target'")
            else:
                logging.error(f"Target column not found in dataframe columns: {list(df.columns)}")
                raise ValueError(f"Target column not found in dataframe columns: {list(df.columns)}")
        
        # Split fitur dan target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Identifikasi fitur numerik dan kategorikal
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")
        
        # Definisikan transformasi untuk preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                # Tambahkan preprocessing untuk fitur kategorikal jika perlu
                # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit preprocessor pada data training
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        logging.info(f"Data preprocessed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        
        # Simpan preprocessor
        os.makedirs('models', exist_ok=True)
        try:
            joblib.dump(preprocessor, 'models/preprocessor.pkl')
            logging.info("Preprocessor saved to models/preprocessor.pkl")
        except Exception as e:
            logging.error(f"Error saving preprocessor: {e}")
            # Jika gagal menyimpan preprocessor, tetap lanjutkan
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error dalam preprocessing data: {e}")
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
    Menjalankan keseluruhan ML pipeline, dari loading data hingga evaluasi model
    
    Args:
        data_path (str): Path ke file CSV data
        model_type (str): Jenis model yang akan dilatih
        
    Returns:
        dict: Dictionary berisi metrik evaluasi model
    """
    try:
        # Mulai timer
        start_time = time.time()
        
        # Setup logging
        setup_logging()
        
        # Step 1: Load data
        logging.info(f"Starting ML pipeline with data from {data_path}")
        df = load_data(data_path)
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Step 3: Train model
        model = train_model(X_train, y_train, model_type)
        
        # Step 4: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log waktu eksekusi
        execution_time = time.time() - start_time
        logging.info(f"Pipeline completed in {execution_time:.2f} seconds")
        
        return metrics
    except Exception as e:
        logging.error(f"Error in ML pipeline: {e}")
        # Return default metrics dalam kasus error
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'error': str(e)
        }

# Main entry point
if __name__ == "__main__":
    run_pipeline() 