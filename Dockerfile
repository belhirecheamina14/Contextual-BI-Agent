# Utiliser une image Python officielle comme base
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier de dépendances et l'installer
# Utiliser un répertoire temporaire pour les dépendances pour optimiser le cache Docker
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Exposer le port sur lequel FastAPI s'exécute
EXPOSE 8000

# Commande de démarrage de l'application
# Utiliser uvicorn pour servir l'application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
