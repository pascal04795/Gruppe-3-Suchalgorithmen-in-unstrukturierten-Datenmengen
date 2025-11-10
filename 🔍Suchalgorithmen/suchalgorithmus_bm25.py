"""
BM25 implementation (simple, dependency-free core).

Usage:
 - Import BM25 and use BM25(corpus).search(query, top_n=10)
 - Or run this file directly. It will try to load `cc_news` via `datasets` and fall back
   to a small sample corpus if that fails.

This file is intended to live in the existing `Suchalgorithmen/` folder.
"""
import math
import re
import time
from collections import Counter
from typing import List, Tuple, Dict
from datasets import load_dataset

betterThanScore = 0.0

dataset = load_dataset("cc_news", split="train[:10%]")
texts = [item['text'] for item in dataset if item['text'] and item['text'].strip()]

def _tokenize(text: str) -> List[str]:
    # simple word tokenizer; keeps only word characters, lowercases
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """Fit BM25 on the given corpus.

        Args:
            corpus: list of documents (strings)
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        # tokenized documents and term frequencies per document
        self.docs_tokens = [ _tokenize(doc) for doc in corpus ]
        self.term_freqs = [ Counter(tokens) for tokens in self.docs_tokens ]
        self.doc_lens = [ len(tokens) for tokens in self.docs_tokens ]
        self.avgdl = sum(self.doc_lens) / self.N if self.N > 0 else 0.0

        # document frequency for each term
        df: Dict[str, int] = {}
        for tf in self.term_freqs:
            for term in tf.keys():
                df[term] = df.get(term, 0) + 1
        self.df = df

        # precompute idf for each term using standard BM25 idf variant
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            # add 1 to make idf positive for terms appearing in many docs
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: str) -> List[float]:
        """Return BM25 scores for all documents for the query."""
        q_terms = _tokenize(query)
        scores = [0.0] * self.N

        if not q_terms or self.N == 0:
            return scores

        for term in q_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.N):
                tf = self.term_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[i] += score

        return scores

    def search(self, query: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """Return top_n documents as (doc_index, score) sorted by score desc."""
        scores = self.get_scores(query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_n]


if __name__ == "__main__":
    
    if not texts:
        raise RuntimeError("No texts found in dataset; falling back")
    

    queries = ["computer", "religion", "space", "sports", "windows"]

    # measure indexing time (BM25 construction)
    index_start = time.time()
    model = BM25(texts)
    index_end = time.time()
    index_duration = index_end - index_start

    # helper: extract best sentence from a document for the query
    def _best_sentence_for_query(doc: str, query: str) -> str:
        # split into sentences
        sents = re.split(r'(?<=[.!?])\s+', doc.strip())
        if not sents:
            return doc[:200].replace('\n', ' ')
        q_terms = set(_tokenize(query))
        # prefer sentence with most query terms, fallback to first sentence
        best = sents[0]
        best_score = 0
        for sent in sents:
            toks = _tokenize(sent)
            # count exact matches of query terms
            score = sum(1 for t in toks if t in q_terms)
            if score > best_score:
                best_score = score
                best = sent
        # return a trimmed snippet
        return best.replace('\n', ' ').strip()[:300]

    # measure query/search time separately
    total_query_time = 0.0
    print(" BM25 Model")
    for q in queries:
        q_start = time.time()

        # compute full scores to count all documents above threshold
        scores = model.get_scores(q)
        count_ge_2 = sum(1 for s in scores if s > betterThanScore)

        top = model.search(q, top_n=5)

        # print results for this query
        print(f"\nQuery: '{q}' ")
        print(f"  Treffer mit score >= {betterThanScore}: {count_ge_2}")
        print("  Top 5 Sätze:")
        for idx, score in top:
            snippet = _best_sentence_for_query(texts[idx], q)
            print(f"    doc={idx:6d} score={score:8.4f} sentence={snippet!r}")

        q_end = time.time()
        q_duration = q_end - q_start
        total_query_time += q_duration
        print(f"  (Zeit für diese Query: {q_duration:.4f} s)")

    end = time.time()
    print(f"\nIndexierungszeit (BM25-Konstruktion): {index_duration:.4f} s")
    print(f"Gesamte Suchzeit (alle Queries): {total_query_time:.4f} s")
    print(f"Gesamtzeit (inkl. Indexierung): {index_duration + total_query_time:.4f} s")

