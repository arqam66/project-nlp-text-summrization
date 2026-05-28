# Summa AI — NLP Text Summarization

Extractive text summarization engine using word-frequency scoring, with ROUGE evaluation and a web-based interface.

## Features

- **Extractive summarization** — scores sentences by word frequency and returns the top-ranked ones
- **ROUGE evaluation** — computes ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, F1)
- **Dataset integration** — loads articles from CNN/DailyMail (or any CSV with `article`/`highlights` columns) for benchmarking
- **Reservoir sampling** — efficiently picks random records from large CSVs without loading them entirely into memory
- **Web UI** — Flask-backed interface with summarization, dataset exploration, and batch evaluation tabs

## Requirements

- Python 3.10+
- [NLTK](https://www.nltk.org/) — tokenization and stopwords
- [Flask](https://flask.palletsprojects.com/) — web server
- [Flask-CORS](https://flask-cors.readthedocs.io/) — cross-origin support

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

NLTK data (`punkt`, `punkt_tab`, `stopwords`) downloads automatically on the first import.

## Usage

### Web UI

```bash
python app.py
```

Open `http://localhost:5000` in a browser.

#### Tabs
| Tab | Description |
|-----|-------------|
| **Summarizer** | Paste any text and generate an extractive summary |
| **Dataset** | Load random articles from the dataset, compare generated vs. reference summaries, and view ROUGE scores |
| **Evaluation** | Run batch evaluation across multiple samples to compute aggregate ROUGE scores |

### CLI / Python

```python
from summarizer import summarize_text, compute_rouge

text = "Your long article text here..."
summary = summarize_text(text, num_sentences=3)
print(summary)

# Compare with a reference
scores = compute_rouge(summary, reference_summary)
print(scores)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/summarize` | Summarize provided text |
| `GET` | `/dataset/info` | Article counts per split |
| `GET` | `/dataset/sample` | Random article(s) from a split |
| `POST` | `/dataset/summarize_sample` | Summarize a dataset article and compare with its reference |
| `POST` | `/dataset/evaluate` | Batch evaluation on random samples |

## Dataset

Place CSV files in `sample_data/` with the following structure:

```
id,article,highlights
```

Expected filenames: `train.csv`, `test.csv`, `validation.csv`.

## Project Structure

```
├── app.py              Flask web server
├── summarizer.py       Core summarization + evaluation logic
├── index.html          Web UI (single-page)
├── requirements.txt    Python dependencies
├── sample_data/        Dataset CSVs (not included in repo)
└── .gitignore
```
