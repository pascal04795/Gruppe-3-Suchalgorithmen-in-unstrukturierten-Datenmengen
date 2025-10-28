#1. Virtuelle Umgebung (.venv) erstellen
#2. Virtuelle Umgebung aktivieren (.venv\Scripts\activate)
#3. Installieren, der Bibliothek (pip install datasets)
#4. ...
import time
import re
import math
from collections import Counter

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def tokenize(text):
    """Very small tokenizer: lowercase and extract word characters."""
    return re.findall(r"\w+", (text or '').lower())


# The simple linear_search was removed per request; the probabilistic
# / model-based search below is adapted to count document matches.


def build_language_models(texts):
    """Build per-document term frequencies and global vocabulary info.

    Returns (docs_tf, doc_lengths, V, vocab_counter)
    """
    docs_tf = []
    vocab = Counter()
    doc_lengths = []
    for text in texts:
        tokens = tokenize(text)
        tf = Counter(tokens)
        docs_tf.append(tf)
        vocab.update(tf)
        doc_lengths.append(len(tokens) if tokens else 0)
    V = len(vocab)
    return docs_tf, doc_lengths, V, vocab


def score_document(query_tokens, tf, doc_len, V, alpha=1.0):
    """Score a document for the query tokens using add-alpha smoothing.

    We compute log P(query | doc) ~= sum(log((tf[w]+alpha)/(doc_len+alpha*V))).
    Documents with length 0 will get a very low score.
    """
    if doc_len == 0:
        # assign very low probability
        return -1e9
    score = 0.0
    denom = doc_len + alpha * V
    for w in query_tokens:
        pw = (tf.get(w, 0) + alpha) / denom
        score += math.log(pw)
    return score


def probabilistic_search(texts, query, docs_tf=None, doc_lengths=None, V=None, alpha=1.0):
    """Count how many documents contain the query tokens.

    For a query with multiple tokens this uses an OR semantics (document
    contains any of the tokens). Returns an integer count.
    """
    if not query or not query.strip():
        return 0
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    if docs_tf is None:
        docs_tf, doc_lengths, V, _ = build_language_models(texts)

    # Count documents that contain at least one query token
    def doc_contains_any(tf, tokens):
        for t in tokens:
            if tf.get(t, 0) > 0:
                return True
        return False

    count = sum(1 for tf in docs_tf if doc_contains_any(tf, query_tokens))
    return count


def _sample_snippet(text, length=200):
    if not text:
        return ''
    s = ' '.join(text.split())
    return (s[:length] + '...') if len(s) > length else s


def main():
    
    dataset = load_dataset("cc_news", split="train[:10%]")
    
    texts = [item['text'] for item in dataset if item.get('text') and item['text'].strip()]
    
    queries = ["computer", "religion", "space", "sports", "windows"]

    # Build document models once for searching
    docs_tf, doc_lengths, V, vocab = build_language_models(texts)

    print(f"Built language models for {len(texts)} documents. Vocabulary size: {V}")

    overall_start = time.time()

    results = []
    for q in queries:
        start = time.time()
        count = probabilistic_search(texts, q, docs_tf=docs_tf, doc_lengths=doc_lengths, V=V)
        end = time.time()

        result = {
            'query': q,
            'num_matches': count,
            'search_time': end - start,
        }
        results.append(result)

    overall_end = time.time()
    total_duration = overall_end - overall_start

    for r in results:
        print("=" * 60)
        print(f"Suchwort: '{r['query']}'")
        print(f"  Treffer: {r['num_matches']}  (Dauer: {r['search_time']:.6f}s)")

    print("=" * 60)
    print(f"Gesamtlaufzeit für alle Suchen: {total_duration:.4f} Sekunden")


if __name__ == '__main__':
    main()
