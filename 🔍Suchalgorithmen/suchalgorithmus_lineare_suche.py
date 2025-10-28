#1. Virtuelle Umgebung (.venv) erstellen
#2. Virtuelle Umgebung aktivieren (.venv\Scripts\activate)
#3. Installieren, der Bibliothek (pip install datasets)
#4. ...
import time
from datasets import load_dataset

def linear_search(texts, query):
    results = []
    for i, text in enumerate(texts):
        if query.lower() in text.lower():
            results.append(i)
    return results

# Lade Wikipedia-Datensatz (10 % der Artikel)
dataset = load_dataset("cc_news", split="train[:10%]")
texts = [item['text'] for item in dataset if item['text'] and item['text'].strip()]

queries = ["computer", "religion", "space", "sports", "windows", ""]

overall_start = time.time()

results = []
for q in queries:
    matches = linear_search(texts, q)
    result = {
        'query': q,
        'num_matches': len(matches),
        'sample_results': matches[:5]
    }
    results.append(result)

overall_end = time.time()
total_duration = overall_end - overall_start

for r in results:
    print(f"Suchwort: {r['query']}")
    print(f"  Treffer: {r['num_matches']}")
    print("-" * 30)


print(f"Gesamtlaufzeit für alle Suchen: {total_duration:.4f} Sekunden")
