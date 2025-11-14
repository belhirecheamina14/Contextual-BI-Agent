from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .api import QueryRequest
from .core.orchestrator import orchestrate_query

app = FastAPI(
    title="Contextual BI Agent API",
    description="API pour l'agent de Business Intelligence conversationnel et contextuel."
)

# Configuration CORS pour permettre l'accès depuis le frontend
origins = [
    "*", # Autoriser toutes les origines pour le PoC
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Contextual BI Agent API is running."}

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Traite la question de l'utilisateur en orchestrant la récupération de données et de contexte.
    """
    try:
        response = await orchestrate_query(request.question)
        return {"response": response}
    except Exception as e:
        return {"response": f"Une erreur interne est survenue : {e}"}

# Montage du répertoire frontend comme un répertoire statique
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
