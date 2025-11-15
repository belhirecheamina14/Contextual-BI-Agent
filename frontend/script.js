document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages');

    // Détermination de l'URL de l'API
    // Si l'URL contient le port exposé (ex: 8000-...), on utilise cette base.
    // Sinon, on utilise localhost:8000 (pour le développement local).
    const API_BASE_URL = window.location.origin.includes('8000') 
        ? window.location.origin.split('/frontend')[0] 
        : 'http://localhost:8000';
    const API_URL = `${API_BASE_URL}/query`;

    // Fonction pour créer un message dans l'interface
    function createMessageElement(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        // Remplacement des sauts de ligne par des balises <br> pour un meilleur affichage
        const formattedText = text.replace(/\n/g, '<br>');
        
        messageDiv.innerHTML = `<p>${formattedText}</p>`;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight; // Scroll vers le bas
        return messageDiv;
    }

    // Fonction pour afficher l'indicateur de chargement
    function showLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'agent-message', 'loading-indicator');
        loadingDiv.id = 'loading-indicator';
        // Utilisation des spans pour l'animation de rebond (définie dans style.css)
        loadingDiv.innerHTML = `<p>L'Agent est en train d'orchestrer la requête... <span></span><span></span><span></span></p>`;
        messagesContainer.appendChild(loadingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Fonction pour masquer l'indicateur de chargement
    function hideLoadingIndicator() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    // Fonction principale pour envoyer la requête
    async function sendMessage() {
        const question = userInput.value.trim();
        if (question === '') return;

        // 1. Afficher le message de l'utilisateur
        createMessageElement(question, 'user');
        userInput.value = ''; // Vider l'input
        sendButton.disabled = true; // Désactiver le bouton pendant le traitement

        // 2. Afficher l'indicateur de chargement
        showLoadingIndicator();

        try {
            // 3. Envoyer la requête à l'API
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            const data = await response.json();

            // 4. Masquer l'indicateur de chargement
            hideLoadingIndicator();

            // 5. Afficher la réponse de l'agent
            if (response.ok) {
                createMessageElement(data.response, 'agent');
            } else {
                createMessageElement(`Erreur de l'API : ${data.response || 'Impossible de contacter le serveur.'}`, 'agent');
            }

        } catch (error) {
            console.error('Erreur lors de la communication avec l\'API:', error);
            hideLoadingIndicator();
            createMessageElement(`Erreur de connexion : Impossible de joindre l'agent BI. Assurez-vous que le backend est démarré sur ${API_BASE_URL}.`, 'agent');
        } finally {
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    // Événements
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Focus initial sur le champ de saisie
    userInput.focus();
});
