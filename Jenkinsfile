pipeline {
    agent any
    
    // Definisi parameter build
    parameters {
        booleanParam(defaultValue: false, description: 'Apakah akan menjalankan tes unit?', name: 'RUN_TESTS')
        booleanParam(defaultValue: true, description: 'Apakah akan melatih model?', name: 'TRAIN_MODEL')
        booleanParam(defaultValue: true, description: 'Apakah akan menjalankan monitoring model?', name: 'RUN_MONITORING')
    }
    
    // Definisi stages pipeline
    stages {
        // Setup lingkungan
        stage('Setup Environment') {
            steps {
                echo 'Setting up environment...'
                
                // Periksa Docker
                bat 'docker --version'
                
                // Dapatkan direktori kerja absolut
                bat 'echo %CD%'
                
                // Periksa struktur direktori
                bat 'dir'
                
                // Buat direktori yang diperlukan jika belum ada
                bat '''
                    if not exist models mkdir models
                    if not exist reports mkdir reports
                    if not exist data mkdir data
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
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                    bat '''
                        docker run --rm -v "%CD%:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python -m pytest test_pipeline.py -v"
                    '''
                }
                // Alternatif jika mengalami masalah dengan %CD%
                // bat '''
                //     docker run --rm -v "D:/AutoML/Jenkins_Automation:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python -m pytest test_pipeline.py -v"
                // '''
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
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python ml_pipeline.py"
                '''
                // Alternatif jika mengalami masalah dengan %CD%
                // bat '''
                //     docker run --rm -v "D:/AutoML/Jenkins_Automation:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python ml_pipeline.py"
                // '''
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
                    docker run --rm -v "%CD%:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python model_monitoring.py"
                '''
                // Alternatif jika mengalami masalah dengan %CD%
                // bat '''
                //     docker run --rm -v "D:/AutoML/Jenkins_Automation:/app" -w /app python:3.10 bash -c "pip install -r requirements.txt && python model_monitoring.py"
                // '''
            }
            post {
                success {
                    // Arsipkan laporan monitoring
                    archiveArtifacts artifacts: 'reports/*.png, reports/*.csv, reports/ALERT_*.txt, reports/*.json', fingerprint: true, allowEmptyArchive: true
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