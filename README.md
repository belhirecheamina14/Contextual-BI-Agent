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
| **Contextualization Engine** | Moteur RAG pour la récupération de contexte métier. | Vector DB (ChromaDB/Simulé) / LLM |
| **Orchestrator** | Gère l'exécution parallèle des tâches de données et de contexte. | Python `asyncio` |

### 🛠️ Installation et Démarrage (PoC)

Ce projet est un Proof of Concept (PoC) et nécessite une clé API pour les modèles de langage (OpenAI ou OpenRouter).

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/belhirecheamina14/Contextual-BI-Agent.git
    cd Contextual-BI-Agent
    ```

2.  **Configuration de l'environnement :**
    Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
    ```
    OPENAI_API_KEY="votre_cle_api_ici"
    # OU
    OPENROUTER_API_KEY="votre_cle_api_ici"
    ```

3.  **Installation des dépendances :**
    ```bash
    pip install -r backend/requirements.txt
    ```

4.  **Démarrage du Backend (API) :**
    ```bash
    uvicorn backend.app.main:app --reload
    ```

5.  **Accès au Frontend :**
    Ouvrez `frontend/index.html` dans votre navigateur.

### 📝 Fichiers Clés

*   `backend/app/core/orchestrator.py` : Le cœur de la logique parallèle.
*   `backend/app/core/data_query.py` : Logique de conversion NL-to-SQL et d'exécution de requête.
*   `backend/app/core/context_rag.py` : Logique de récupération de contexte RAG.
*   `backend/app/data/sales_data.csv` : Données BI simulées.
*   `backend/app/data/kpi_definitions.txt` : Documentation métier simulée.
