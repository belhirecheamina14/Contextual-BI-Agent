import asyncio
import os
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv

from .data_query import execute_data_query, get_data_schema
from .context_rag import retrieve_context

load_dotenv()

# Initialisation du client OpenAI (utilise OPENAI_API_KEY ou OPENROUTER_API_KEY de l'environnement)
# Si OPENROUTER_API_KEY est défini, il sera utilisé. Sinon, OPENAI_API_KEY.
client = OpenAI()

# Modèle à utiliser pour la génération de code et de réponse
LLM_MODEL = "gpt-4.1-mini" # Modèle rapide et performant

async def generate_pandas_code(question: str) -> str:
    """
    Utilise le LLM pour traduire la question en code Pandas.
    """
    data_schema = get_data_schema()
    
    system_prompt = f"""
    Vous êtes un traducteur de langage naturel en code Pandas.
    Votre tâche est de générer un fragment de code Python/Pandas qui répond à la question de l'utilisateur en utilisant le DataFrame 'df'.
    Le schéma de données est le suivant : {data_schema}.
    Le code doit être une expression Python valide qui peut être évaluée (ex: df['Sales'].sum()).
    NE PAS inclure 'import pandas as pd' ou 'df = pd.read_csv(...)'.
    NE PAS inclure de commentaires ou d'explications.
    NE RETOURNER QUE le code Pandas.
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.0
        )
        # Nettoyage du code généré (suppression des backticks si le LLM les ajoute)
        code = response.choices[0].message.content.strip()
        if code.startswith("```") and code.endswith("```"):
            code = code.split('\n', 1)[-1].rsplit('\n', 1)[0]
        return code.strip()
    except Exception as e:
        return f"Erreur de génération de code Pandas: {e}"

async def get_data_result(question: str) -> str:
    """
    Génère le code Pandas et exécute la requête de données.
    """
    pandas_code = await generate_pandas_code(question)
    if "Erreur" in pandas_code:
        return f"Échec de la traduction NL-to-Pandas: {pandas_code}"
        
    data_result = execute_data_query(pandas_code)
    return data_result

async def get_context_result(question: str) -> str:
    """
    Récupère le contexte métier pertinent via le moteur RAG.
    """
    # La fonction retrieve_context est synchrone, nous l'exécutons dans un thread pool
    return await asyncio.to_thread(retrieve_context, question)

async def orchestrate_query(question: str) -> str:
    """
    Orchestre l'exécution parallèle de la requête de données et de la récupération de contexte.
    """
    # Exécution parallèle des deux tâches
    data_task = asyncio.create_task(get_data_result(question))
    context_task = asyncio.create_task(get_context_result(question))
    
    # Attente des résultats
    data_result, context_result = await asyncio.gather(data_task, context_task)
    
    # Génération de la réponse finale
    final_answer = await generate_final_answer(question, data_result, context_result)
    
    return final_answer

async def generate_final_answer(question: str, data_result: str, context_result: str) -> str:
    """
    Utilise le LLM pour synthétiser la réponse finale en utilisant les données et le contexte.
    """
    system_prompt = f"""
    Vous êtes un Agent de Business Intelligence Contextuel.
    Votre tâche est de synthétiser une réponse complète et professionnelle à la question de l'utilisateur.
    Vous devez utiliser le 'Résultat de la Requête de Données' pour fournir la réponse factuelle.
    Vous devez utiliser le 'Contexte Métier' pour expliquer les définitions, les règles de calcul ou les implications.
    Si le 'Résultat de la Requête de Données' indique une erreur, vous devez l'expliquer poliment à l'utilisateur.
    Si le 'Contexte Métier' est "Aucun contexte métier pertinent trouvé.", vous devez vous concentrer uniquement sur la réponse factuelle.
    
    Question de l'utilisateur: {question}
    
    Résultat de la Requête de Données:
    ---
    {data_result}
    ---
    
    Contexte Métier:
    ---
    {context_result}
    ---
    
    Votre réponse doit être rédigée en français, claire et structurée.
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur de génération de la réponse finale: {e}"
