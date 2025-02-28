pipeline {
    agent {
        docker {
            image 'python:3.10' // Gunakan image Python resmi
            args '-v ${WORKSPACE}:/app' // Mount workspace ke direktori /app di container
        }
    }
    
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
                
                // Periksa versi Python dan pip
                sh 'python --version'
                sh 'pip --version'
                
                // Install dependensi
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }
        
        // Unit testing
        stage('Run Tests') {
            when {
                expression { params.RUN_TESTS == true }
            }
            steps {
                echo 'Running unit tests...'
                sh 'python -m pytest test_pipeline.py -v'
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
                sh 'python ml_pipeline.py'
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
                sh 'python model_monitoring.py'
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