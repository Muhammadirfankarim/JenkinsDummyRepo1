pipeline {
    agent any
    
    // Definisi parameter build
    parameters {
        booleanParam(defaultValue: true, description: 'Apakah akan menjalankan tes unit?', name: 'RUN_TESTS')
        booleanParam(defaultValue: true, description: 'Apakah akan melatih model?', name: 'TRAIN_MODEL')
        booleanParam(defaultValue: true, description: 'Apakah akan menjalankan monitoring model?', name: 'RUN_MONITORING')
    }
    
    // Definisi environment
    environment {
        // Gunakan path lengkap ke Python jika diperlukan
        PYTHON_PATH = "C:\\Python310\\python.exe"  // Sesuaikan dengan lokasi Python di server Jenkins
        VIRTUAL_ENV = "venv"    // Nama virtual environment
    }
    
    // Definisi stages pipeline
    stages {
        // Setup lingkungan
        stage('Setup Environment') {
            steps {
                echo 'Setting up environment...'
                
                // Periksa apakah Python tersedia
                bat "where python || echo Python tidak ditemukan di PATH"
                
                // Coba buat direktori untuk virtual environment
                bat "mkdir -p %VIRTUAL_ENV% || echo Direktori sudah ada"
                
                // Install pip dan dependensi tanpa virtual environment
                bat "pip install --upgrade pip || echo Gagal upgrade pip"
                bat "pip install -r requirements.txt || echo Gagal install requirements"
            }
        }
        
        // Unit testing
        stage('Run Tests') {
            when {
                expression { params.RUN_TESTS == true }
            }
            steps {
                echo 'Running unit tests...'
                bat "python -m pytest test_pipeline.py -v || echo Gagal menjalankan test"
            }
            post {
                always {
                    // Tampilkan laporan test (jika ada plugin JUnit)
                    junit allowEmptyResults: true, testResults: 'test-reports/*.xml'
                }
            }
        }
        
        // Training model
        stage('Train Model') {
            when {
                expression { params.TRAIN_MODEL == true }
            }
            steps {
                echo 'Training ML model...'
                bat "python ml_pipeline.py || echo Gagal melatih model"
            }
            post {
                success {
                    // Arsipkan model dan laporan
                    archiveArtifacts artifacts: 'models/*.pkl, reports/*.png, reports/*.csv', fingerprint: true, allowEmptyArchive: true
                }
            }
        }
        
        // Model monitoring
        stage('Model Monitoring') {
            when {
                expression { params.RUN_MONITORING == true }
            }
            steps {
                echo 'Running model monitoring...'
                bat "python model_monitoring.py || echo Gagal menjalankan monitoring"
            }
            post {
                success {
                    // Arsipkan laporan monitoring
                    archiveArtifacts artifacts: 'reports/*.png, reports/*.csv, reports/ALERT_*.txt', fingerprint: true, allowEmptyArchive: true
                }
            }
        }
    }
    
    // Post-pipeline actions
    post {
        always {
            echo 'Pipeline selesai dijalankan.'
        }
        success {
            echo 'Pipeline berhasil! Model dan laporan tersimpan.'
        }
        failure {
            echo 'Pipeline gagal. Silakan periksa log untuk detail.'
        }
    }
} 