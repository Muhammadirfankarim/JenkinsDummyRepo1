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
    
    def test_load_data(self):
        """
        Pengujian fungsi load_data
        """
        # Test load data (akan membuat data dummy jika tidak ada)
        df = load_data()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('target' in df.columns)
        self.assertGreater(len(df), 0)
    
    def test_preprocess_data(self):
        """
        Pengujian fungsi preprocess_data
        """
        # Load data
        df = load_data()
        
        # Preprocess data
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
        
        # Pastikan scaler telah disimpan
        self.assertTrue(os.path.exists('models/scaler.pkl'))
    
    def test_train_model(self):
        """
        Pengujian fungsi train_model
        """
        # Load dan preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Latih model
        model = train_model(X_train, y_train)
        
        # Pastikan model valid
        self.assertIsNotNone(model)
        
        # Pastikan model telah disimpan
        self.assertTrue(os.path.exists('models/model.pkl'))
        
        # Memuat model dan periksa prediksi
        loaded_model = joblib.load('models/model.pkl')
        preds = loaded_model.predict(X_test[:5])
        self.assertEqual(len(preds), 5)
    
    def test_evaluate_model(self):
        """
        Pengujian fungsi evaluate_model
        """
        # Load dan preprocess data
        df = load_data()
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
        
        # Periksa plot feature importance
        self.assertTrue(os.path.exists('reports/feature_importance.png'))
        
        # Periksa file metrics
        metrics_files = [f for f in os.listdir('reports') if f.startswith('metrics_') and f.endswith('.csv')]
        self.assertGreater(len(metrics_files), 0)
    
    def test_full_pipeline(self):
        """
        Pengujian full pipeline
        """
        # Jalankan full pipeline
        metrics = run_pipeline()
        
        # Pastikan metrics valid
        self.assertIsNotNone(metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Periksa apakah files yang diperlukan ada
        self.assertTrue(os.path.exists('models/model.pkl'))
        self.assertTrue(os.path.exists('models/scaler.pkl'))
        self.assertTrue(os.path.exists('reports/feature_importance.png'))

if __name__ == '__main__':
    # Pastikan direktori test-reports ada
    if not os.path.exists('test-reports'):
        os.makedirs('test-reports')
    
    # Jalankan unittest dengan XMLRunner untuk laporan JUnit
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports')) 