# Python and environment setup
PYTHON = python3
ENV_NAME = venv
REQUIREMENTS = requirements.txt

# Help function to display all available targets and their descriptions
help:
	@echo "Makefile for automating tasks related to MLOps with Metricbeat, Kibana, and Elasticsearch"
	@echo ""
	@echo "Available Targets:"
	@echo "  lint               - Run code quality checks using black, flake8, pylint, and bandit"
	@echo "  prepare            - Prepare data for the model"
	@echo "  train              - Train the model"
	@echo "  evaluate           - Evaluate the model"
	@echo "  predict            - Predict with the trained model"
	@echo "  test               - Run tests without displaying warnings"
	@echo "  run-web            - Start the Flask web app"
	@echo "  run-api            - Start the FastAPI server"
	@echo "  open-swagger       - Open Swagger UI in the browser"
	@echo "  run_mlflow         - Start the MLflow UI"
	@echo "  test-mlflow        - Test the MLflow server"
	@echo "  build              - Build the Docker image"
	@echo "  push               - Push the Docker image to the registry"
	@echo "  run                - Run FastAPI and MLflow in the container"
	@echo "  docker             - Run build, push, and run tasks sequentially"
	@echo "  start-containers   - Start all containers with Docker Compose"
	@echo "  start-metricbeat   - Start the Metricbeat service"
	@echo "  stop-containers    - Stop all containers with Docker Compose"
	@echo "  all                - Run all steps (prepare, train, evaluate, predict, test, lint)"
	@echo "  start_airflow      - Run start_airflow airflow"
	@echo ""

# Linting (Code quality check)
lint:
	@echo "Vérification de la qualité du code..."
	@$(ENV_NAME)/bin/black .
	@$(ENV_NAME)/bin/flake8 .
	@$(ENV_NAME)/bin/pylint main.py
	@$(ENV_NAME)/bin/bandit -r .

# Prepare data
prepare:
	@echo "Preparing data..."
	@bash -c ". $(ENV_NAME)/bin/activate && $(PYTHON) main.py --prepare"

# Train model
train:
	@echo "Entraînement du modèle..."
	@bash -c ". $(ENV_NAME)/bin/activate && $(PYTHON) main.py --train"

# Evaluate model (Renamed for clarity)
evaluate:
	@echo "Evaluating model..."
	@bash -c ". $(ENV_NAME)/bin/activate && $(PYTHON) main.py --evaluate"

# Predict with the model
predict:
	@echo "Predicting with the model..."
	@bash -c ". $(ENV_NAME)/bin/activate && $(PYTHON) main.py --predict"

# Run all tests without displaying warnings
test:
	@echo "Running tests..."
	@$(ENV_NAME)/bin/pytest -p no:warnings --maxfail=1 --disable-warnings -q

# Run Flask web app
run-web:
	@echo "Starting Flask web app on port 5000..."
	@$(ENV_NAME)/bin/python web_app.py --port 5000 &  # Run Flask in the background
	@echo "Starting FastAPI server on port 8000..."
	@bash -c ". $(ENV_NAME)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000" &  # Run FastAPI in the background
	@echo "Flask app is running at http://localhost:5000"
	@echo "FastAPI app is running at http://localhost:8000/docs"
	@wait  # Wait for both processes to finish (this ensures the Makefile waits until both are done)


# Run FastAPI server
run-api:
	@echo "Démarrage du serveur FastAPI..."
	@bash -c ". $(ENV_NAME)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000"

# Makefile target to open Swagger UI in the browser
open-swagger:
	@echo "Starting FastAPI server..."
	@bash -c ". $(ENV_NAME)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000" & xdg-open http://127.0.0.1:8000/docs


# Start MLflow UI
run_mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
	@echo "Opening MLflow UI in browser..."
	xdg-open http://localhost:5000 || google-chrome http://localhost:5000

# Test the MLflow server
test-mlflow:
	@echo "Testing MLflow server..."
	curl -X 'GET' 'http://127.0.0.1:5000' -H 'accept: application/json'

# Docker commands

IMAGE_NAME = erijj/erij_mazouz_4ds8_churn_mlops_test
DOCKER_TAG = latest

# Build Docker image
build:
	docker build -t $(IMAGE_NAME):$(DOCKER_TAG) .

# Push Docker image to registry
push:
	docker push $(IMAGE_NAME):$(DOCKER_TAG)

# Run FastAPI and MLflow in the Docker container
run:
	@echo "Running FastAPI and MLflow..."
	@docker run -d -p 8000:8000 -p 5000:5000 $(IMAGE_NAME):$(DOCKER_TAG) bash -c "uvicorn app:app --reload --host 0.0.0.0 --port 8000 & mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &"

# Combine the docker targets in one rule by running them sequentially
docker: build push run

# Makefile for automating tasks related to MLOps with Metricbeat, Kibana, and Elasticsearch

# Start all containers (Kibana, Elasticsearch, Prometheus, etc.) using Docker Compose
start-containers:
	docker-compose up -d  # Start containers in detached mode

# Start Metricbeat service (to collect Docker container metrics)
start-metricbeat:
	sudo service metricbeat start  # Start Metricbeat to collect metrics

# Stop all containers using Docker Compose
stop-containers:
	docker-compose down  # Stop all containers

start_airflow:
	airflow webserver -p 8081 &
	airflow scheduler &




# Run all steps (prepare, train, evaluate, predict, test, lint)
all: prepare train evaluate predict test lint
