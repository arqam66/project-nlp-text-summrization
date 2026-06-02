import math
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

def _get_content_words(text):
    """Return list of lowercase content words (no stopwords, no punctuation)."""
    stop_words = set(stopwords.words('english'))
    return [
        w.lower() for w in word_tokenize(text)
        if w.isalpha() and w.lower() not in stop_words
    ]


def _jaccard_similarity(a_words, b_words):
    """Jaccard similarity between two sets of content words."""
    if not a_words or not b_words:
        return 0
    a_set, b_set = set(a_words), set(b_words)
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0


CUE_WORDS = {
    'concluded', 'conclusion', 'demonstrated', 'discovered', 'essential',
    'finding', 'findings', 'found', 'identified', 'important', 'key',
    'notably', 'proposed', 'proves', 'reported', 'result', 'results',
    'revealed', 'reveals', 'showed', 'shows', 'significant', 'suggests',
    'summary', 'therefore', 'thus', 'vital',
}


def summarize_text(text, num_sentences=3):
    """
    Meaning-focused extractive summarizer.

    Scoring:
      - Keyword importance via log-IDF (rare meaningful words weighted higher)
      - Keyword coverage (sentences covering more key topics score higher)
      - Position bias (first sentences capture the lede)
      - Cue-word boost (findings, conclusions, results)
      - MMR diversity (no two similar sentences selected)
      - Adaptive count based on text length
    """
    if not text:
        return ""

    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    if not sentences:
        return text

    # Adaptive sentence count — at least 2, at most 6, based on text size
    word_count = len(text.split())
    auto_k = max(2, min(6, word_count // 80))
    k = min(num_sentences or auto_k, len(sentences))
    if k < 1:
        k = 1
    if len(sentences) <= k:
        return ' '.join(sentences)

    n = len(sentences)

    # ── 1. Build vocab ───────────────────────────────────────────────────
    word_tf = {}
    word_sf = {}
    sent_words = []

    for sent in sentences:
        cw = _get_content_words(sent)
        sent_words.append(cw)
        seen = set()
        for w in cw:
            word_tf[w] = word_tf.get(w, 0) + 1
            if w not in seen:
                word_sf[w] = word_sf.get(w, 0) + 1
                seen.add(w)

    if not word_tf:
        return sentences[0]

    max_tf = max(word_tf.values())
    n_sents = n

    # ── 2. Keyword importance (TF-logIDF) ────────────────────────────────
    keyword_weight = {}
    for w, tf in word_tf.items():
        idf = math.log((1 + n_sents) / (1 + word_sf[w])) + 1
        keyword_weight[w] = (tf / max_tf) * idf

    # Top 15 keywords define the doc's main topics
    top_keywords = set(
        w for w, _ in heapq.nlargest(15, keyword_weight.items(), key=lambda x: x[1])
    )

    # ── 3. Score each sentence ────────────────────────────────────────────
    raw_scores = []
    for i, sent in enumerate(sentences):
        cw = sent_words[i]
        if not cw:
            raw_scores.append(0)
            continue

        kw_score = sum(keyword_weight.get(w, 0) for w in cw) / len(cw)

        coverage = len(top_keywords & set(cw)) / len(top_keywords) if top_keywords else 0

        pos_score = 1.0 / (1 + i)

        cue_score = 1.5 if any(w in CUE_WORDS for w in cw) else 0.0

        # Bonus for sentences containing multiple top keywords
        kw_count = sum(1 for w in cw if w in top_keywords)
        density = kw_count / len(cw) if cw else 0

        score = (kw_score * 0.45 + coverage * 0.20 + pos_score * 0.15 + cue_score * 0.10 + density * 0.10)
        raw_scores.append(score)

    # ── 4. Pick sentences with MMR ───────────────────────────────────────
    selected = []
    candidates = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_val = -9999.0
        for j in candidates:
            sim = 0.0
            if selected:
                sim = max(
                    _jaccard_similarity(sent_words[j], sent_words[s])
                    for s in selected
                )
            mmr = 0.7 * raw_scores[j] - 0.3 * sim
            if mmr > best_val:
                best_val = mmr
                best_idx = j
        if best_idx != -1:
            selected.append(best_idx)
            candidates.remove(best_idx)

    # ── 5. Present by original order for coherent flow ───────────────────
    selected.sort()
    summary = ' '.join(sentences[i] for i in selected)
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
