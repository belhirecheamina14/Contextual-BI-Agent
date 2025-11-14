# Contextual BI Agent

## Agent de Business Intelligence Conversationnel et Contextuel

Le **Contextual BI Agent** est une solution innovante qui vise à combler la lacune entre les données brutes de Business Intelligence (BI) et la compréhension métier des utilisateurs non-techniques. Il utilise l'intelligence artificielle pour fournir des réponses factuelles et **contextualisées** aux questions posées en langage naturel.

### 🚀 Proposition de Valeur

Traditionnellement, les outils de BI nécessitent une expertise pour naviguer et interpréter les données. Notre agent simplifie l'accès à l'information en :

1.  **Traduisant le Langage Naturel en Requêtes de Données (NL-to-SQL/Pandas)**.
2.  **Récupérant des Données Factuelles** (ex: Chiffre d'affaires, Taux de conversion).
3.  **Enrichissant la Réponse avec le Contexte Métier** (ex: Définition des KPI, règles de calcul) via un moteur RAG (Retrieval-Augmented Generation).
4.  **Orchestrant en Parallèle** la récupération des données et du contexte pour une réponse rapide et complète.

### 🏗️ Architecture

L'architecture est modulaire et basée sur des microservices :

| Composant | Rôle | Technologie Clé |
| :--- | :--- | :--- |
| **Frontend** | Interface utilisateur conversationnelle. | HTML/CSS/JS |
| **API Gateway** | Point d'entrée unique (FastAPI). | Python / FastAPI |
| **Data Query Engine** | Exécution des requêtes de données. | Pandas / LLM (NL-to-SQL) |
| **Contextualization Engine** | Moteur RAG pour la récupération de contexte métier. | Vector DB (Simulé) / LLM |
| **Orchestrator** | Gère l'exécution parallèle des tâches de données et de contexte. | Python `asyncio` |

### 🛠️ Déploiement et Utilisation (Recommandé : Docker)

Ce projet est conçu pour être facilement déployé à l'aide de Docker et Docker Compose.

#### Prérequis

*   Docker et Docker Compose installés.
*   Une clé API pour un modèle de langage (OpenAI ou OpenRouter).

#### 1. Configuration

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/belhirecheamina14/Contextual-BI-Agent.git
    cd Contextual-BI-Agent
    ```

2.  **Clé API :**
    Éditez le fichier `docker-compose.yml` et remplacez `votre_cle_api_ici` par votre clé API réelle (pour `OPENAI_API_KEY` ou `OPENROUTER_API_KEY`).

#### 2. Démarrage de l'Application

Lancez l'application avec Docker Compose :
```bash
docker-compose up --build
```
L'application sera accessible à l'adresse `http://localhost:8000/frontend/index.html`.

#### 3. Utilisation (Mode Développement)

Si vous préférez exécuter l'application sans Docker (mode développement) :

1.  **Installation des dépendances :**
    ```bash
    pip install -r backend/requirements.txt
    ```

2.  **Configuration de l'environnement :**
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
    ```
    OPENAI_API_KEY="votre_cle_api_ici"
    # OU
    OPENROUTER_API_KEY="votre_cle_api_ici"
    ```

3.  **Démarrage du Backend (API) :**
    ```bash
    uvicorn backend.app.main:app --reload
    ```

4.  **Accès au Frontend :**
    Ouvrez `frontend/index.html` dans votre navigateur.

### 📝 Fichiers Clés

*   `Dockerfile` : Définition de l'image Docker.
*   `docker-compose.yml` : Configuration pour le déploiement.
*   `backend/app/core/orchestrator.py` : Le cœur de la logique parallèle.
*   `backend/app/data/sales_data.csv` : Données BI simulées.
*   `backend/app/data/kpi_definitions.txt` : Documentation métier simulée.
