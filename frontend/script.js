const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const API_URL = 'http://127.0.0.1:8000/query'; // URL de l'API FastAPI

function addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    
    // Remplacement des sauts de ligne par des balises <br> pour un meilleur affichage
    const formattedText = text.replace(/\n/g, '<br>');
    
    messageDiv.innerHTML = `<p>${formattedText}</p>`;
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll vers le bas
}

function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('loading-message');
    loadingDiv.id = 'loading-agent';
    loadingDiv.innerHTML = '<p>L\'Agent est en train d\'orchestrer la requête...</p>';
    chatWindow.appendChild(loadingDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeLoadingMessage() {
    const loadingDiv = document.getElementById('loading-agent');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

async function sendMessage() {
    const question = userInput.value.trim();
    if (question === '') return;

    // 1. Afficher le message de l'utilisateur
    addMessage('user', question);
    userInput.value = '';
    
    // 2. Afficher le message de chargement
    addLoadingMessage();

    try {
        // 3. Envoyer la requête à l'API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();
        
        // 4. Supprimer le message de chargement
        removeLoadingMessage();

        // 5. Afficher la réponse de l'Agent
        addMessage('agent', data.response);

    } catch (error) {
        console.error('Erreur lors de la communication avec l\'API:', error);
        removeLoadingMessage();
        addMessage('agent', `Désolé, une erreur est survenue lors de la communication avec le serveur : ${error.message}. Veuillez vérifier que le backend est bien démarré sur ${API_URL}.`);
    }
}

// Événements
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
