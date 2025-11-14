import pandas as pd
import os

# Le chemin est relatif au répertoire racine du projet (Contextual-BI-Agent)
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")

def get_data_schema() -> str:
    """Retourne le schéma des données pour aider le LLM à générer la requête."""
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        return str(df.head(1).to_dict())
    except Exception as e:
        return f"Erreur lors de la lecture du schéma de données : {e}"

def execute_data_query(pandas_code: str) -> str:
    """
    Exécute le code pandas généré par le LLM et retourne le résultat.

    Args:
        pandas_code: Le code Python/Pandas à exécuter.

    Returns:
        Le résultat de l'exécution sous forme de chaîne de caractères.
    """
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # Exécution du code pandas dans un environnement sécurisé (simulé ici)
        # NOTE: Dans un environnement de production, l'exécution de code arbitraire est dangereuse.
        # Un mécanisme de validation strict ou un moteur de requête dédié serait nécessaire.
        
        # Pour ce PoC, nous allons simplement évaluer le code qui utilise 'df'
        # Exemple de code attendu: "df[df['Region'] == 'North']['Sales'].sum()"
        
        # Création d'un dictionnaire d'exécution pour isoler l'exécution
        exec_globals = {'df': df, 'pd': pd}
        exec_locals = {}
        
        # Exécution du code
        exec(f"result = {pandas_code}", exec_globals, exec_locals)
        
        result = exec_locals.get('result', 'Aucun résultat retourné.')
        
        return str(result)
    except Exception as e:
        return f"Erreur lors de l'exécution de la requête Pandas: {e}"
