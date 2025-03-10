# Étape 1 : Choisir l'image de base
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans l'image
WORKDIR /app

# Étape 3 : Copier les fichiers du projet dans le conteneur
COPY . /app

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Exposer le port pour l'API FastAPI
EXPOSE 8000

# Étape 6 : Démarrer le serveur FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

