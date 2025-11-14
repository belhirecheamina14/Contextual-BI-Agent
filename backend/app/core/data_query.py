import pandas as pd
import os
import json
from .safe_query_engine import SafeQueryExecutor

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
    Exécute le code pandas généré par le LLM et retourne le résultat en utilisant SafeQueryExecutor.

    Args:
        pandas_code: Le code Python/Pandas à exécuter.

    Returns:
        Le résultat de l'exécution sous forme de chaîne de caractères JSON.
    """
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        executor = SafeQueryExecutor(df)
        result_dict = executor.execute(pandas_code)

        if result_dict.get("success"):
            # Formate le résultat en une chaîne JSON pour la cohérence
            return json.dumps(result_dict["result"], indent=2)
        else:
            error_message = result_dict.get('error', 'Une erreur inconnue est survenue.')
            return f"Erreur lors de l'exécution de la requête Pandas: {error_message}"

    except FileNotFoundError:
        return f"Erreur: Le fichier de données est introuvable à {DATA_FILE_PATH}"
    except Exception as e:
        # Attrape d'autres erreurs potentielles (ex: pannes de SafeQueryExecutor)
        return f"Erreur critique dans l'exécution de la requête : {e}"
