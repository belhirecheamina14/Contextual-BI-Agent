import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Le chemin est relatif au répertoire racine du projet (Contextual-BI-Agent)
CONTEXT_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kpi_definitions.txt")

def load_context_data() -> list[str]:
    """Charge les définitions de KPI et les divise en chunks pour la recherche."""
    try:
        with open(CONTEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        # Division par double saut de ligne pour simuler des "chunks" de documents
        chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        return chunks
    except Exception as e:
        print(f"Erreur lors du chargement des données de contexte : {e}")
        return []

def retrieve_context(question: str) -> str:
    """
    Simule la récupération de contexte (RAG) en utilisant la similarité TF-IDF.

    Args:
        question: La question de l'utilisateur.

    Returns:
        Le contexte le plus pertinent sous forme de chaîne de caractères.
    """
    chunks = load_context_data()
    if not chunks:
        return "Aucun contexte métier disponible."

    # Ajout de la question aux documents pour le calcul TF-IDF
    documents = chunks + [question]
    
    # Calcul TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # La dernière ligne est le vecteur de la question
    question_vector = tfidf_matrix[-1]
    
    # Calcul de la similarité cosinus entre la question et tous les chunks
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix[:-1]).flatten()
    
    # Récupération de l'index du chunk le plus similaire
    most_similar_index = cosine_similarities.argmax()
    
    # Seuil de pertinence (arbitraire pour la simulation)
    if cosine_similarities[most_similar_index] > 0.1:
        return chunks[most_similar_index]
    else:
        return "Aucun contexte métier pertinent trouvé."
