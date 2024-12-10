document.addEventListener('DOMContentLoaded', () => {
    const ui = {
        select: document.getElementById('conversation-select'),
        modelSelect: document.getElementById('model-select'),
        analyzeBtn: document.getElementById('analyze-btn'),
        loading: document.getElementById('loading'),
        resultsContainer: document.getElementById('results-container'),
        mainConcerns: document.getElementById('main-concerns'),
        nextSteps: document.getElementById('next-steps'),
        transcriptEl: document.getElementById('transcript'),
        confusionList: document.getElementById('confusion-list'),
        tabBtns: document.querySelectorAll('.tab-btn'),
        filterCheckboxes: document.querySelectorAll('.filter-checkbox'),
        insightsEl: document.getElementById('insights'),
        clinicianInsightsEl: document.getElementById('clinician-insights'),

        // New elements for upload
        uploadAudioBtn: document.getElementById('upload-audio-btn'),
        audioConversationId: document.getElementById('audio-conversation-id'),
        audioFile: document.getElementById('audio-file'),

        uploadTextBtn: document.getElementById('upload-text-btn'),
        textConversationId: document.getElementById('text-conversation-id'),
        textInput: document.getElementById('text-input')
    };

    let globalData = {
        analysis: null,
        transcriptLines: [],
        entities: []
    };

    // Handle tab switching
    ui.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            ui.tabBtns.forEach(b => b.classList.remove('border-blue-600', 'border-b-2', 'text-gray-800'));
            btn.classList.add('border-b-2', 'border-blue-600', 'text-gray-800');

            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
            const activeTab = btn.dataset.tab;
            document.getElementById(`${activeTab}-tab`).classList.remove('hidden');
        });
    });

    // Handle analyze
    ui.analyzeBtn.addEventListener('click', async () => {
        const conversationId = ui.select.value;
        const modelName = ui.modelSelect.value;
        ui.loading.classList.remove('hidden');
        ui.resultsContainer.classList.add('hidden');

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: conversationId, model_name: modelName })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            globalData.analysis = data.analysis;
            renderResults(data);
            ui.loading.classList.add('hidden');
            ui.resultsContainer.classList.remove('hidden');
        } catch (e) {
            console.error(e);
            alert('Error: ' + e.message);
            ui.loading.classList.add('hidden');
        }
    });

    ui.filterCheckboxes.forEach(cb => {
        cb.addEventListener('change', applyFilters);
    });

    // Handle audio upload
    ui.uploadAudioBtn.addEventListener('click', async () => {
        const convId = ui.audioConversationId.value.trim();
        const file = ui.audioFile.files[0];
        if (!convId || !file) {
            alert("Please provide a conversation ID and select an audio file.");
            return;
        }

        const formData = new FormData();
        formData.append('conversation_id', convId);
        formData.append('file', file);

        try {
            const res = await fetch('/upload_audio', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.error) {
                alert("Error uploading audio: " + data.error);
            } else {
                alert("Audio uploaded successfully! Refresh the page to see the conversation in the dropdown.");
            }
        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        }
    });

    // Handle text upload
    ui.uploadTextBtn.addEventListener('click', async () => {
        const convId = ui.textConversationId.value.trim();
        const text = ui.textInput.value.trim();
        if (!convId || !text) {
            alert("Please provide a conversation ID and paste some text.");
            return;
        }

        try {
            const res = await fetch('/upload_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: convId, text })
            });
            const data = await res.json();
            if (data.error) {
                alert("Error uploading text: " + data.error);
            } else {
                alert("Text uploaded successfully! Refresh the page to see the conversation in the dropdown.");
            }
        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        }
    });

    function renderResults(data) {
        const { analysis, transcript } = data;
        renderExecutiveSummary(analysis.summary);
        globalData.entities = analysis.entities;
        globalData.transcriptLines = transcript.split('\n').filter(l => l.trim());

        const highlightedTranscript = highlightTranscript(globalData.transcriptLines, globalData.entities);
        renderTranscript(highlightedTranscript);

        renderEntities(globalData.entities);
        renderConfusions(analysis.highlighted_confusions);
        renderInsights(globalData.entities);
        renderClinicianInsights(analysis.clinician_insights);

        ui.tabBtns.forEach(b => b.classList.remove('border-b-2', 'border-blue-600', 'text-gray-800'));
        document.querySelector('[data-tab="findings"]').classList.add('border-b-2', 'border-blue-600', 'text-gray-800');
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
        document.getElementById('findings-tab').classList.remove('hidden');

        applyFilters();
    }

    function renderExecutiveSummary(summary) {
        ui.mainConcerns.innerHTML = summary.main_concerns
            .map(c => `<li class="text-sm text-gray-700 leading-tight">${c}</li>`)
            .join('');
        ui.nextSteps.innerHTML = summary.critical_next_steps
            .map(s => `<li class="text-sm text-gray-700 leading-tight">${s}</li>`)
            .join('');
    }

    function renderTranscript(lines) {
        ui.transcriptEl.innerHTML = lines.map((turn, i) => `
        <div class="turn p-2 rounded hover:bg-gray-50" data-turn="${i}">${turn}</div>
      `).join('');
    }

    function renderEntities(entities) {
        const grouped = { condition: [], medication: [], instruction: [], follow_up: [] };
        entities.forEach(e => {
            grouped[e.type].push(e);
        });

        for (const [type, ents] of Object.entries(grouped)) {
            const container = document.querySelector(`#${type}s .space-y-2`);
            if (!container) continue;
            container.innerHTML = ents.map(e => entityCard(e)).join('');
        }
    }

    function renderClinicianInsights(insights) {
        ui.clinicianInsightsEl.innerHTML = insights.map(
            insight => `<li class="leading-tight list-item list-disc ml-4">${insight}</li>`
        ).join('');
    }

    function entityCard(e) {
        const turns = e.related_turns.join(',');
        const displayText = e.patient_explanation || e.text || '(No explanation)';

        return `
        <div class="p-4 bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow cursor-pointer relative"
          onclick="highlightTurns('${turns}')">
          <div class="flex items-start justify-between">
            <div class="pr-6">
              <div class="flex items-center space-x-2 mb-1">
                <span class="text-sm font-medium text-gray-800">${e.text || '(No text)'}</span>
                <span class="px-2 py-0.5 text-xs rounded-full ${getTypeColor(e.type)}">${e.type}</span>
              </div>
              <p class="text-sm text-gray-600 line-clamp-2">${displayText}</p>
            </div>
            <button class="text-blue-600 text-xs hover:text-blue-700 focus:outline-none"
              onclick="event.stopPropagation();toggleMoreInfo(this)">More info</button>
          </div>
          <div class="more-info hidden mt-3 pt-3 border-t border-gray-100">
            ${moreInfoContent(e)}
          </div>
        </div>
      `;
    }

    window.toggleMoreInfo = function (btn) {
        const parent = btn.closest('.p-4');
        const moreInfo = parent.querySelector('.more-info');
        moreInfo.classList.toggle('hidden');
    };

    function moreInfoContent(e) {
        let info = `<div class="space-y-1 text-xs text-gray-700">`;
        info += `<div><strong>Type:</strong> ${e.type}</div>`;
        if (e.dosage) info += `<div><strong>Dosage:</strong> ${e.dosage}</div>`;
        if (e.first_mention_turn !== null) info += `<div><strong>First Mention Turn:</strong> ${e.first_mention_turn}</div>`;
        if (e.needs_clarification) info += `<div class="text-amber-600">Needs clarification</div>`;
        info += `</div>`;
        return info;
    }

    function getTypeColor(type) {
        const colors = {
            condition: 'bg-red-100 text-red-800',
            medication: 'bg-green-100 text-green-800',
            instruction: 'bg-yellow-100 text-yellow-800',
            follow_up: 'bg-blue-100 text-blue-800'
        };
        return colors[type] || 'bg-gray-100 text-gray-800';
    }

    function renderConfusions(confusions) {
        ui.confusionList.innerHTML = confusions.map(c => `
        <div class="p-2 bg-amber-50 rounded text-sm">${c}</div>
      `).join('');
    }

    function renderInsights(entities) {
        const counts = { condition: 0, medication: 0, instruction: 0, follow_up: 0 };
        entities.forEach(e => { counts[e.type] = (counts[e.type] || 0) + 1; });

        ui.insightsEl.innerHTML = `
        <div class="space-y-2">
          <div class="text-sm font-medium text-gray-700">Aggregate Insights:</div>
          <div>Conditions: ${counts.condition}</div>
          <div>Medications: ${counts.medication}</div>
          <div>Instructions: ${counts.instruction}</div>
          <div>Follow-ups: ${counts.follow_up}</div>
        </div>
      `;
    }

    function applyFilters() {
        const filters = {};
        ui.filterCheckboxes.forEach(cb => {
            filters[cb.dataset.type] = cb.checked;
        });

        document.querySelectorAll('.entity-group').forEach(g => {
            const t = g.getAttribute('data-type');
            g.style.display = filters[t] ? '' : 'none';
        });
    }

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

    function highlightTranscript(lines, entities) {
        const byTurn = {};
        entities.forEach(e => {
            if (e.first_mention_turn !== null && e.text) {
                byTurn[e.first_mention_turn] = byTurn[e.first_mention_turn] || [];
                byTurn[e.first_mention_turn].push(e);
            }
        });

        return lines.map((line, i) => {
            if (!byTurn[i]) return line;
            let highlightedLine = line;
            for (const ent of byTurn[i]) {
                const { text, patient_explanation } = ent;
                if (!text) continue;
                const idx = highlightedLine.toLowerCase().indexOf(text.toLowerCase());
                if (idx >= 0) {
                    const before = highlightedLine.slice(0, idx);
                    const match = highlightedLine.slice(idx, idx + text.length);
                    const after = highlightedLine.slice(idx + text.length);
                    const tooltip = patient_explanation || text;
                    highlightedLine = `${before}<span class="entity-inline" title="${tooltip}">${match}</span>${after}`;
                }
            }
            return highlightedLine;
        });
    }
});
