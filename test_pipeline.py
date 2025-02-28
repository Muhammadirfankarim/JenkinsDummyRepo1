import unittest
import os
import pandas as pd
import numpy as np
from ml_pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline
import joblib
import xmlrunner  # Menambahkan import untuk xmlrunner

class TestMLPipeline(unittest.TestCase):
    """
    Test case untuk machine learning pipeline
    """
    def setUp(self):
        """
        Setup untuk test case
        """
        # Pastikan direktori yang diperlukan ada
        for dir_name in ['data', 'models', 'reports']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        
        # Buat data dummy untuk test jika belum ada
        if not os.path.exists('data/test.csv'):
            # Buat data dengan kedua format target (y dan target)
            self.create_test_data()
    
    def create_test_data(self):
        """
        Membuat data dummy untuk testing
        """
        # Data dengan kolom target bernama 'target'
        df_target = pd.DataFrame({
            'feature_0': np.random.randn(10),
            'feature_1': np.random.randn(10),
            'target': np.random.randint(0, 2, 10)
        })
        os.makedirs('data', exist_ok=True)
        df_target.to_csv('data/test.csv', index=False)
        
        # Data dengan kolom target bernama 'y'
        df_y = pd.DataFrame({
            'age': [30, 35, 40, 45],
            'job': ['admin', 'blue-collar', 'entrepreneur', 'admin'],
            'marital': ['married', 'single', 'married', 'divorced'],
            'education': ['primary', 'secondary', 'tertiary', 'secondary'],
            'balance': [1000, 2000, 3000, 4000],
            'housing': ['yes', 'no', 'yes', 'no'],
            'loan': ['no', 'no', 'yes', 'yes'],
            'y': ['no', 'yes', 'no', 'yes']
        })
        df_y.to_csv('data/test_bank.csv', index=False)
    
    def test_load_data(self):
        """
        Pengujian fungsi load_data
        """
        # Test load data (akan membuat data dummy jika tidak ada)
        df = load_data('data/test.csv')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('target' in df.columns)
        self.assertGreater(len(df), 0)
        
        # Test load data dengan format bank - ubah ekspektasi untuk mendukung target sebagai kolom
        df_bank = load_data('data/test_bank.csv')
        self.assertIsNotNone(df_bank)
        self.assertIsInstance(df_bank, pd.DataFrame)
        # Gunakan ini daripada hanya mencari 'y'
        self.assertTrue(any(col in df_bank.columns for col in ['y', 'target']), 
                       f"Kolom target tidak ditemukan dalam {df_bank.columns}")
    
    def test_preprocess_data(self):
        """
        Pengujian fungsi preprocess_data
        """
        # Test dengan data yang memiliki kolom target 'target'
        df = load_data('data/test.csv')
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Pastikan hasil preprocessing valid
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        
        # Periksa bentuk data
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(len(y_test), X_test.shape[0])
        
        # Test dengan data yang memiliki kolom target 'y'
        df_bank = load_data('data/test_bank.csv')
        X_train_bank, X_test_bank, y_train_bank, y_test_bank = preprocess_data(df_bank)
        
        # Pastikan hasil preprocessing valid
        self.assertIsNotNone(X_train_bank)
        self.assertIsNotNone(X_test_bank)
        self.assertIsNotNone(y_train_bank)
        self.assertIsNotNone(y_test_bank)
        
        # Pastikan preprocessor telah disimpan
        self.assertTrue(os.path.exists('models/preprocessor.pkl'))
        self.assertTrue(os.path.exists('models/scaler.pkl'))
    
    def test_train_model(self):
        """
        Pengujian fungsi train_model
        """
        # Load dan preprocess data
        df = load_data('data/test.csv')
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Latih model
        model = train_model(X_train, y_train)
        
        # Pastikan model valid
        self.assertIsNotNone(model)
        
        # Pastikan model telah disimpan
        self.assertTrue(os.path.exists('models/model.pkl'))
        
        # Memuat model dan periksa prediksi
        loaded_model = joblib.load('models/model.pkl')
        preds = loaded_model.predict(X_test[:1])
        self.assertEqual(len(preds), 1)
    
    def test_evaluate_model(self):
        """
        Pengujian fungsi evaluate_model
        """
        # Load dan preprocess data
        df = load_data('data/test.csv')
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Latih model
        model = train_model(X_train, y_train)
        
        # Evaluasi model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Pastikan metrics valid
        self.assertIsNotNone(metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Periksa file metrik
        metrics_files = [f for f in os.listdir('reports') if f.startswith('metrics_') and f.endswith('.csv')]
        self.assertGreater(len(metrics_files), 0)
    
    def test_full_pipeline(self):
        """
        Pengujian full pipeline
        """
        # Jalankan full pipeline dengan data yang memiliki kolom 'target'
        metrics = run_pipeline('data/test.csv')
        
        # Pastikan metrics valid
        self.assertIsNotNone(metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Periksa apakah files yang diperlukan ada
        self.assertTrue(os.path.exists('models/model.pkl'))
        self.assertTrue(os.path.exists('models/preprocessor.pkl'))
        self.assertTrue(os.path.exists('models/scaler.pkl'))

if __name__ == '__main__':
    # Pastikan direktori test-reports ada
    if not os.path.exists('test-reports'):
        os.makedirs('test-reports')
    
    # Jalankan unittest dengan XMLRunner untuk laporan JUnit
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports')) 