pipeline {
    agent any

    environment {
        PYTHON     = 'C:\\Users\\prasa\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
        PIP        = 'C:\\Users\\prasa\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m pip'
        VENV_DIR   = '.venv'
        OUTPUT_DIR = 'outputs'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '--- Checking out source from GitHub ---'
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                echo '--- Creating virtual environment and installing dependencies ---'
                bat """
                    "${PYTHON}" -m venv ${VENV_DIR}
                    ${VENV_DIR}\\Scripts\\python.exe -m pip install --upgrade pip
                    ${VENV_DIR}\\Scripts\\pip.exe install -r requirements.txt
                """
            }
        }

        stage('Lint') {
            steps {
                echo '--- Running syntax check on all Python files ---'
                bat """
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile carshare_mlops/data.py
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile carshare_mlops/model.py
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile carshare_mlops/monitoring.py
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile carshare_mlops/tracking.py
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile api.py
                    ${VENV_DIR}\\Scripts\\python.exe -m py_compile main.py
                """
            }
        }

        stage('Unit Tests - Pipeline') {
            steps {
                echo '--- Running Pytest pipeline tests (data, model, evidently) ---'
                bat """
                    ${VENV_DIR}\\Scripts\\python.exe -m pytest tests/test_pipeline.py -v ^
                        --tb=short ^
                        --junitxml=outputs/junit_pipeline.xml
                """
            }
            post {
                always {
                    junit 'outputs/junit_pipeline.xml'
                }
            }
        }

        stage('Unit Tests - API') {
            steps {
                echo '--- Running HTTPX-based FastAPI tests ---'
                bat """
                    ${VENV_DIR}\\Scripts\\python.exe -m pytest tests/test_api.py -v ^
                        --tb=short ^
                        --junitxml=outputs/junit_api.xml
                """
            }
            post {
                always {
                    junit 'outputs/junit_api.xml'
                }
            }
        }

        stage('Run ML Pipeline') {
            steps {
                echo '--- Training model, generating Evidently drift report, logging to MLflow ---'
                bat """
                    ${VENV_DIR}\\Scripts\\python.exe main.py
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                echo '--- Archiving outputs ---'
                archiveArtifacts artifacts: 'outputs/**', allowEmptyArchive: true
            }
        }
    }

    post {
        success {
            echo '========================================='
            echo ' BUILD SUCCESSFUL - All stages passed'
            echo '========================================='
        }
        failure {
            echo '========================================='
            echo ' BUILD FAILED - Check console output'
            echo '========================================='
        }
        always {
            echo '--- Pipeline complete ---'
        }
    }
}
