pipeline {
    agent any
    
    // Definisi parameter build
    parameters {
        booleanParam(defaultValue: true, description: 'Apakah akan menjalankan tes unit?', name: 'RUN_TESTS')
        booleanParam(defaultValue: true, description: 'Apakah akan melatih model?', name: 'TRAIN_MODEL')
        booleanParam(defaultValue: true, description: 'Apakah akan menjalankan monitoring model?', name: 'RUN_MONITORING')
    }
    
    // Definisi stages pipeline
    stages {
        // Setup lingkungan
        stage('Setup Environment') {
            steps {
                echo 'Setting up environment...'
                
                // Jalankan Docker untuk instalasi dependensi
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 pip install -r requirements.txt
                '''
                
                // Periksa instalasi
                bat '''
                    echo Instalasi paket berhasil
                    dir
                '''
            }
        }
        
        // Unit testing
        stage('Run Tests') {
            when {
                expression { params.RUN_TESTS == true }
            }
            steps {
                echo 'Running unit tests...'
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 python -m pytest test_pipeline.py -v
                '''
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
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 python ml_pipeline.py
                '''
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
                bat '''
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 python model_monitoring.py
                '''
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