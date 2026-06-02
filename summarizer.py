import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import csv
import os
import random

# Download necessary NLTK data
for resource, path in [('punkt', 'tokenizers/punkt'),
                       ('punkt_tab', 'tokenizers/punkt_tab'),
                       ('stopwords', 'corpora/stopwords')]:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


# ── Sample dataset helpers ──────────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

def _load_csv_rows(filename, max_rows=None):
    """Load rows from a Sample CSV file."""
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def _count_csv_records(filepath):
    """Count the actual number of CSV records (not raw lines) in a file."""
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            count += 1
    return count


def get_random_sample(split='test', count=1):
    """Return random article(s) from the dataset with their reference summaries.
    
    Uses reservoir sampling to efficiently pick random records from large CSVs
    without loading everything into memory.
    
    Args:
        split: 'train', 'test', or 'validation'
        count: number of samples to return
    
    Returns:
        list of dicts with keys: id, article, highlights
    """
    filename = f'{split}.csv'
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        return []

    # Count actual CSV records (handles multi-line fields correctly)
    total = _count_csv_records(filepath)
    if total <= 0:
        return []

    # Pick random record indices
    count = min(count, total)
    indices = set(random.sample(range(total), count))

    results = []
    with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in indices:
                results.append({
                    'id': row.get('id', ''),
                    'article': row.get('article', ''),
                    'highlights': row.get('highlights', '')
                })
                if len(results) == count:
                    break
    return results


def get_dataset_info():
    """Return info about available dataset splits."""
    info = {}
    for split in ['train', 'test', 'validation']:
        filepath = os.path.join(DATASET_DIR, f'{split}.csv')
        if os.path.exists(filepath):
            info[split] = _count_csv_records(filepath)
    return info


# ── Core summarization ─────────────────────────────────────────────────────

def summarize_text(text, num_sentences=3):
    """Extractive summarization using word-frequency scoring."""
    if not text:
        return ""

    # Preprocessing
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Calculate word frequency
    word_frequencies = {}
    for word in words:
        if word.isalnum() and word not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Normalize frequency
    if not word_frequencies:
        return text[:100] + "..." if len(text) > 100 else text
        
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # Sentence scoring
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    # Get summary — clamp to available sentences
    if num_sentences < 1:
        num_sentences = 1
    num_sentences = min(num_sentences, len(sentence_scores))
    if num_sentences == 0:
        return text

    top_sentences = set(heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get))
    summary_sentences = [sent for sent in sentences if sent in top_sentences]
    summary = ' '.join(summary_sentences)
    
    return summary


# ── Evaluation (ROUGE-like) ────────────────────────────────────────────────

def _get_ngrams(tokens, n):
    """Get n-grams from a list of tokens."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_rouge(generated, reference):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    Returns dict with precision, recall, f1 for each metric.
    """
    gen_tokens = word_tokenize(generated.lower())
    ref_tokens = word_tokenize(reference.lower())

    if not gen_tokens or not ref_tokens:
        return {'rouge1': {'p': 0, 'r': 0, 'f1': 0},
                'rouge2': {'p': 0, 'r': 0, 'f1': 0},
                'rougeL': {'p': 0, 'r': 0, 'f1': 0}}

    def _rouge_n(n):
        gen_ngrams = _get_ngrams(gen_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        if not gen_ngrams or not ref_ngrams:
            return {'p': 0, 'r': 0, 'f1': 0}
        gen_set = {}
        for ng in gen_ngrams:
            gen_set[ng] = gen_set.get(ng, 0) + 1
        ref_set = {}
        for ng in ref_ngrams:
            ref_set[ng] = ref_set.get(ng, 0) + 1
        overlap = 0
        for ng, count in ref_set.items():
            overlap += min(count, gen_set.get(ng, 0))
        precision = overlap / len(gen_ngrams) if gen_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'p': round(precision, 4), 'r': round(recall, 4), 'f1': round(f1, 4)}

    # ROUGE-L (Longest Common Subsequence)
    def _lcs_length(x, y):
        m, n = len(x), len(y)
        # Use space-optimized LCS for large inputs
        if m > 500 or n > 500:
            # Approximate with shorter sequences
            x = x[:500]
            y = y[:500]
            m, n = len(x), len(y)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    lcs = _lcs_length(gen_tokens, ref_tokens)
    rouge_l_p = lcs / len(gen_tokens) if gen_tokens else 0
    rouge_l_r = lcs / len(ref_tokens) if ref_tokens else 0
    rouge_l_f1 = 2 * rouge_l_p * rouge_l_r / (rouge_l_p + rouge_l_r) if (rouge_l_p + rouge_l_r) > 0 else 0

    return {
        'rouge1': _rouge_n(1),
        'rouge2': _rouge_n(2),
        'rougeL': {'p': round(rouge_l_p, 4), 'r': round(rouge_l_r, 4), 'f1': round(rouge_l_f1, 4)}
    }


def evaluate_on_dataset(num_samples=10, num_sentences=3, split='test'):
    """Evaluate the summarizer on random samples from the specified dataset split.
    
    Returns average ROUGE scores and individual results.
    """
    samples = get_random_sample(split, num_samples)
    if not samples:
        return {'error': 'No dataset samples available'}

    results = []
    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for sample in samples:
        generated = summarize_text(sample['article'], num_sentences)
        reference = sample['highlights']
        scores = compute_rouge(generated, reference)
        
        results.append({
            'id': sample['id'],
            'article_preview': sample['article'][:200] + '...',
            'generated_summary': generated,
            'reference_summary': reference,
            'scores': scores
        })
        totals['rouge1'] += scores['rouge1']['f1']
        totals['rouge2'] += scores['rouge2']['f1']
        totals['rougeL'] += scores['rougeL']['f1']

    n = len(results)
    averages = {
        'rouge1_f1': round(totals['rouge1'] / n, 4),
        'rouge2_f1': round(totals['rouge2'] / n, 4),
        'rougeL_f1': round(totals['rougeL'] / n, 4)
    }

    return {
        'num_samples': n,
        'averages': averages,
        'results': results
    }


# ── CLI test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Show dataset info
    info = get_dataset_info()
    if info:
        print("Sample Dataset:")
        for split, count in info.items():
            print(f"  {split}: {count:,} articles")
        print()

    # Test with a dataset sample
    samples = get_random_sample('test', 1)
    if samples:
        sample = samples[0]
        print("=" * 60)
        print("ARTICLE (first 300 chars):")
        print(sample['article'][:300])
        print()
        print("REFERENCE SUMMARY:")
        print(sample['highlights'])
        print()
        generated = summarize_text(sample['article'], 3)
        print("GENERATED SUMMARY:")
        print(generated)
        print()
        scores = compute_rouge(generated, sample['highlights'])
        print("ROUGE SCORES:")
        for metric, vals in scores.items():
            print(f"  {metric}: P={vals['p']:.4f}  R={vals['r']:.4f}  F1={vals['f1']:.4f}")
    else:
        # Fallback to built-in sample
        sample_text = """
        Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and 
        human language, in particular how to program computers to process and analyze large 
        amounts of natural language data. The goal is a computer capable of "understanding" 
        the contents of documents, including the contextual nuances of the language within them. 
        The technology can then accurately extract information and insights contained in the 
        documents as well as categorize and organize the documents themselves. Challenges in 
        natural language processing frequently involve speech recognition, natural-language 
        understanding, and natural-language generation.
        """
        print("Original Text:\n", sample_text)
        print("\nSummary:\n", summarize_text(sample_text, 2))
