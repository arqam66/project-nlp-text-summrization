document.addEventListener('DOMContentLoaded', () => {
    // ── Tab switching ─────────────────────────────────────────────────────
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });

    // ── Tab 1: Summarizer ─────────────────────────────────────────────────
    const summarizeBtn = document.getElementById('summarize-btn');
    const inputText = document.getElementById('input-text');
    const numSentences = document.getElementById('num-sentences');
    const resultSection = document.getElementById('result-section');
    const summaryOutput = document.getElementById('summary-output');
    const loader = document.getElementById('loader');

    summarizeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const sentences = numSentences.value;

        if (!text) {
            alert('Please enter some text to summarize.');
            return;
        }

        summarizeBtn.disabled = true;
        loader.style.display = 'inline-block';
        resultSection.classList.remove('visible');

        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, num_sentences: parseInt(sentences) }),
            });
            const data = await response.json();

            if (data.summary) {
                summaryOutput.textContent = data.summary;
                resultSection.classList.add('visible');
            } else {
                alert('Error: ' + (data.error || 'Failed to generate summary.'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Could not connect to the server. Make sure app.py is running.');
        } finally {
            summarizeBtn.disabled = false;
            loader.style.display = 'none';
        }
    });

    // ── Tab 2: Dataset Explorer ───────────────────────────────────────────
    const loadSampleBtn = document.getElementById('load-sample-btn');
    const datasetLoader = document.getElementById('dataset-loader');
    const datasetResultSection = document.getElementById('dataset-result-section');
    const datasetArticle = document.getElementById('dataset-article');
    const datasetReference = document.getElementById('dataset-reference');
    const datasetGenerated = document.getElementById('dataset-generated');
    const rougeScores = document.getElementById('rouge-scores');
    const datasetInfoEl = document.getElementById('dataset-info');

    // Load dataset info on page load
    async function loadDatasetInfo() {
        try {
            const res = await fetch('/dataset/info');
            const info = await res.json();
            if (info && !info.error) {
                datasetInfoEl.innerHTML = Object.entries(info)
                    .map(([split, count]) =>
                        `<span class="stat-chip"><strong>${split}:</strong> ${count.toLocaleString()} articles</span>`
                    ).join('');
            }
        } catch (e) {
            console.error('Failed to load dataset info:', e);
        }
    }
    loadDatasetInfo();

    loadSampleBtn.addEventListener('click', async () => {
        const split = document.getElementById('dataset-split').value;
        const sentences = parseInt(document.getElementById('dataset-num-sentences').value);

        loadSampleBtn.disabled = true;
        datasetLoader.style.display = 'inline-block';

        try {
            // Step 1: Load a random sample
            const sampleRes = await fetch(`/dataset/sample?split=${split}&count=1`);
            const sampleData = await sampleRes.json();

            if (!sampleData.samples || sampleData.samples.length === 0) {
                alert('No samples found in the dataset.');
                return;
            }

            const sample = sampleData.samples[0];

            // Step 2: Summarize and compare
            const sumRes = await fetch('/dataset/summarize_sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    article: sample.article,
                    highlights: sample.highlights,
                    num_sentences: sentences
                }),
            });
            const sumData = await sumRes.json();

            // Display results
            datasetArticle.textContent = sample.article;
            datasetReference.textContent = sample.highlights;
            datasetGenerated.textContent = sumData.generated_summary;

            if (sumData.rouge_scores) {
                rougeScores.innerHTML = renderScoreCards(sumData.rouge_scores);
            }

            datasetResultSection.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to load sample. Make sure the server is running.');
        } finally {
            loadSampleBtn.disabled = false;
            datasetLoader.style.display = 'none';
        }
    });

    // ── Tab 3: Evaluation ─────────────────────────────────────────────────
    const runEvalBtn = document.getElementById('run-eval-btn');
    const evalLoader = document.getElementById('eval-loader');
    const evalResultSection = document.getElementById('eval-result-section');
    const evalAverages = document.getElementById('eval-averages');
    const evalResults = document.getElementById('eval-results');

    runEvalBtn.addEventListener('click', async () => {
        const numSamples = parseInt(document.getElementById('eval-samples').value);
        const numSent = parseInt(document.getElementById('eval-sentences').value);
        const evalSplit = document.getElementById('eval-split').value;

        runEvalBtn.disabled = true;
        evalLoader.style.display = 'inline-block';
        evalResultSection.style.display = 'none';

        try {
            const res = await fetch('/dataset/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_samples: numSamples,
                    num_sentences: numSent,
                    split: evalSplit
                }),
            });
            const data = await res.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Render average scores
            evalAverages.innerHTML = `
                <div class="score-item">
                    <div class="score-label">ROUGE-1 F1</div>
                    <div class="score-value">${(data.averages.rouge1_f1 * 100).toFixed(1)}%</div>
                    <div class="score-details">${data.num_samples} samples</div>
                </div>
                <div class="score-item">
                    <div class="score-label">ROUGE-2 F1</div>
                    <div class="score-value">${(data.averages.rouge2_f1 * 100).toFixed(1)}%</div>
                    <div class="score-details">${data.num_samples} samples</div>
                </div>
                <div class="score-item">
                    <div class="score-label">ROUGE-L F1</div>
                    <div class="score-value">${(data.averages.rougeL_f1 * 100).toFixed(1)}%</div>
                    <div class="score-details">${data.num_samples} samples</div>
                </div>
            `;

            // Render individual results
            evalResults.innerHTML = data.results.map((r, i) => `
                <div class="eval-item">
                    <div class="eval-item-header">
                        <span class="eval-item-num">#${i + 1}</span>
                        <div class="eval-item-scores">
                            <span class="eval-score-tag">R1: ${(r.scores.rouge1.f1 * 100).toFixed(1)}%</span>
                            <span class="eval-score-tag">R2: ${(r.scores.rouge2.f1 * 100).toFixed(1)}%</span>
                            <span class="eval-score-tag">RL: ${(r.scores.rougeL.f1 * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <p class="eval-item-preview">${r.article_preview}</p>
                </div>
            `).join('');

            evalResultSection.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Evaluation failed. Make sure the server is running.');
        } finally {
            runEvalBtn.disabled = false;
            evalLoader.style.display = 'none';
        }
    });

    // ── Helpers ────────────────────────────────────────────────────────────
    function renderScoreCards(scores) {
        return Object.entries(scores).map(([metric, vals]) => `
            <div class="score-item">
                <div class="score-label">${metric.toUpperCase()}</div>
                <div class="score-value">${(vals.f1 * 100).toFixed(1)}%</div>
                <div class="score-details">P: ${(vals.p * 100).toFixed(1)}% · R: ${(vals.r * 100).toFixed(1)}%</div>
            </div>
        `).join('');
    }
});
