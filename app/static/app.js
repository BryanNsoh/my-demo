document.getElementById('analyze-btn').addEventListener('click', () => {
    const convoSelect = document.getElementById('conversation-select');
    const conversation_id = convoSelect.value;
    const loading = document.getElementById('loading');
    const resultsEl = document.getElementById('results');

    resultsEl.textContent = "";
    loading.style.display = "block";

    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id })
    })
        .then(response => response.json())
        .then(data => {
            loading.style.display = "none";
            resultsEl.textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => {
            loading.style.display = "none";
            resultsEl.textContent = "Error: " + err;
        });
});
