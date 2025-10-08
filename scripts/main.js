// Get references to the HTML elements
const emailInput = document.getElementById('emailInput');
const checkButton = document.getElementById('checkButton');

// --- NEW: Get references to Modal elements ---
const resultModal = document.getElementById('result-modal');
const modalText = document.getElementById('modal-text');
const closeButton = document.getElementById('close-button');

// Function to show the modal
function showModal(message, color) {
    modalText.textContent = message;
    modalText.style.color = color;
    resultModal.style.display = 'block';
}

// Function to hide the modal
function hideModal() {
    resultModal.style.display = 'none';
}

// --- NEW: Add event listeners to close the modal ---
closeButton.addEventListener('click', hideModal);
resultModal.addEventListener('click', (event) => {
    // Hide modal if user clicks on the dark overlay (outside the content box)
    if (event.target === resultModal) {
        hideModal();
    }
});

// Main event listener for the "Check spam" button
checkButton.addEventListener('click', () => {
    const emailText = emailInput.value;

    if (emailText.trim() === "") {
        showModal("Please enter an email to check.", '#e74c3c');
        return;
    }

    const dataToSend = {
        message: emailText
    };

    // Use fetch() to send data to your Django backend
    fetch('https://spam-email-detector-1-843g.onrender.com/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
    .then(response => response.json())
    .then(data => {
        // --- UPDATED: Show the result in the modal ---
        if (data.prediction === 'spam') {
            showModal("Prediction: 🚨 It's a Spam Mail", '#e74c3c'); // Red
        } else {
            showModal("Prediction: ✅ It's a Ham Mail", '#2ecc71'); // Green
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showModal("Could not connect to the server.", '#e74c3c');
    });
});