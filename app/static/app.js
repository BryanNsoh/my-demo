document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        select: document.getElementById('conversation-select'),
        analyzeBtn: document.getElementById('analyze-btn'),
        loading: document.getElementById('loading'),
        resultsContainer: document.getElementById('results-container'),
        mainConcerns: document.getElementById('main-concerns'),
        nextSteps: document.getElementById('next-steps'),
        transcriptEl: document.getElementById('transcript'),
        confusionList: document.getElementById('confusion-list'),
        tabBtns: document.querySelectorAll('.tab-btn'),
    };

    ui.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            ui.tabBtns.forEach(b => b.classList.remove('border-b-2', 'border-blue-600'));
            btn.classList.add('border-b-2', 'border-blue-600');

            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
            const activeTab = btn.dataset.tab;
            document.getElementById(`${activeTab}-tab`).classList.remove('hidden');
        });
    });

    ui.analyzeBtn.addEventListener('click', async () => {
        const conversationId = ui.select.value;
        ui.loading.classList.remove('hidden');
        ui.resultsContainer.classList.add('hidden');

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: conversationId })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            renderResults(data);
            ui.loading.classList.add('hidden');
            ui.resultsContainer.classList.remove('hidden');
        } catch (e) {
            console.error(e);
            alert('Error: ' + e.message);
            ui.loading.classList.add('hidden');
        }
    });

    function renderResults(data) {
        const { analysis, transcript } = data;
        renderExecutiveSummary(analysis.summary);
        renderTranscript(transcript);
        renderEntities(analysis.entities);
        renderConfusions(analysis.highlighted_confusions);

        // Default to findings tab
        ui.tabBtns.forEach(b => b.classList.remove('border-b-2', 'border-blue-600'));
        document.querySelector('[data-tab="findings"]').classList.add('border-b-2', 'border-blue-600');
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
        document.getElementById('findings-tab').classList.remove('hidden');
    }

    function renderExecutiveSummary(summary) {
        ui.mainConcerns.innerHTML = summary.main_concerns.map(c => `<li>${c}</li>`).join('');
        ui.nextSteps.innerHTML = summary.critical_next_steps.map(s => `<li>${s}</li>`).join('');
    }

    function renderTranscript(text) {
        const turns = text.split('\n').filter(l => l.trim());
        ui.transcriptEl.innerHTML = turns.map((turn, i) => `
        <div class="turn p-2" data-turn="${i}">${turn}</div>
      `).join('');
    }

    function renderEntities(entities) {
        const grouped = { condition: [], medication: [], instruction: [], follow_up: [] };
        entities.forEach(e => {
            grouped[e.type].push(e);
        });

        Object.entries(grouped).forEach(([type, ents]) => {
            const container = document.querySelector(`#${type}s .space-y-2`);
            if (!container) return;
            container.innerHTML = ents.map(e => entityCard(e)).join('');
        });
    }

    function entityCard(e) {
        const turns = e.related_turns.join(',');
        return `
        <div class="p-2 bg-white border rounded hover:bg-gray-50 cursor-pointer"
          onclick="highlightTurns('${turns}')">
          <div class="font-medium text-sm">${e.text || '(No text provided)'}</div>
          ${e.patient_explanation ? `<div class="text-xs text-gray-600">${e.patient_explanation}</div>` : ''}
          ${e.needs_clarification ? `<div class="text-xs text-amber-600">Needs clarification</div>` : ''}
        </div>
      `;
    }

    function renderConfusions(confusions) {
        ui.confusionList.innerHTML = confusions.map(c => `
        <div class="p-2 bg-amber-50 rounded text-sm">${c}</div>
      `).join('');
    }

    // Global function to highlight turns
    window.highlightTurns = function (turns) {
        document.querySelectorAll('.turn.highlighted').forEach(el => {
            el.classList.remove('bg-blue-100', 'highlighted');
        });
        const arr = turns.split(',').map(n => parseInt(n, 10));
        arr.forEach(i => {
            const t = document.querySelector(`[data-turn="${i}"]`);
            if (t) {
                t.classList.add('bg-blue-100', 'highlighted');
                t.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    };
});
