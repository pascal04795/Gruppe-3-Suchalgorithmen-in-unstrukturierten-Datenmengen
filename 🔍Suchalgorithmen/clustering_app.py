import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import datasets # Für die Spielplatz-Daten

# --- App-Konfiguration ---
st.set_page_config(page_title="Clustering-Vergleich", layout="wide")
APP_VERSION = "v3.2.3"
st.title("Vergleich von Clustering-Algorithmen")

# --- Globale Hilfsfunktionen für Seite 1 ---

@st.cache_resource
def get_vectorizer_and_texts():
    """Lädt NUR Texte und den Vektorisierer für die Suche."""
    with st.spinner(f"Lade 'train[:10%]' von cc_news..."):
        dataset = load_dataset("cc_news", split="train[:10%]")
        texts = [item['text'] for item in dataset if item['text'] and item['text'].strip()]
    
    with st.spinner(f"Vektorisiere {len(texts)} Artikel (TF-IDF)..."):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(texts)
    
    return texts, vectorizer, X_tfidf

@st.cache_resource(show_spinner=False) # Spinner wird manuell gesteuert
def run_kmeans_clustering(_X_tfidf, k):
    """Führt K-Means-Clustering aus und speichert es im Cache."""
    start_time = time.perf_counter()
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(_X_tfidf)
    end_time = time.perf_counter()
    cluster_time = (end_time - start_time)
    return model, labels, cluster_time

def linear_search(texts_list, query):
    """Die Baseline-Suche."""
    results_indices = []
    query_lower = query.lower()
    for i, text in enumerate(texts_list):
        if query_lower in text.lower():
            results_indices.append(i)
    return results_indices

# --- Globale Hilfsfunktionen für Seite 2 ---

@st.cache_data
def lade_spielplatz_daten(name):
    """Lädt die 2D-Spielplatz-Datensätze."""
    if name == "Monde (Moons)":
        X, y = datasets.make_moons(n_samples=200, noise=0.05, random_state=42)
    elif name == "Kreise (Circles)":
        X, y = datasets.make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=42)
    else: # Blobs
        X, y = datasets.make_blobs(n_samples=200, centers=4, cluster_std=0.6, random_state=42)
    X = StandardScaler().fit_transform(X)
    return X

def plotte_spielplatz(X, labels, algo_name):
    """Erstellt ein Matplotlib-Diagramm"""
    fig, ax = plt.subplots(figsize=(4, 3)) 
    is_noise = (labels == -1); is_core = ~is_noise
    if np.any(is_noise):
        ax.scatter(X[is_noise, 0], X[is_noise, 1], c='gray', marker='x', label='Rauschen')
    scatter = ax.scatter(X[is_core, 0], X[is_core, 1], c=labels[is_core], cmap='viridis', s=40, alpha=0.8)
    ax.set_title(f"Ergebnis für {algo_name}")
    ax.set_xlabel(""); ax.set_ylabel("")
    if len(set(labels[is_core])) > 0:
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
    return fig

# --- #################### APP-LAYOUT #################### ---

tab1, tab2 = st.tabs([
    "Performance-Vergleich (Suche)", 
    "Demo der Algorithmen"
])

# --- #################### TAB 1: SUCHE #################### ---
with tab1:
    st.header("Anwendung: Suche in unstrukturierten Datenmengen")
    st.info("**Ziel**: Testen, wie K-Means als 'Filter' zur Beschleunigung einer Suche auf 70.000 Texten taugt.")
    
    # Lade die Basisdaten für die Suche
    try:
        texts, vectorizer, X_tfidf = get_vectorizer_and_texts()
        
        st.header("Schritt 1: K-Means-Vorbereitung")
        
        with st.expander("Warum K-Means?"):
            st.info("""
            **Warum K-Means funktioniert:**
            * K-Means ist nicht nur schnell, sondern speichert vor allem einen "Mittelpunkt" (Centroid) für jeden Cluster.
            * Dadurch kann es mit seiner `.predict()`-Funktion einen neuen Suchbegriff (z.B. "computer") 
            sofort dem relevantesten Cluster (z.B. "Cluster 5 - Thema: software, data, ...") zuordnen.
            """)
            st.warning("""
            **Warum fehlt Hierarchisches Clustering?**
            * Es ist rechnerisch unmöglich. 
            * Für 70.000 Artikel müsste es eine `70k x 70k` Distanzmatrix berechnen, was ca. **400 GB RAM** erfordern würde. Es ist für die *Vorbereitung* großer Datensätze ungeeignet.
            
            **Warum fehlt DBSCAN?**
            * Es ist für die *Suche* ungeeignet. DBSCAN ist gut darin, Ausreißer (Rauschen) zu finden. Bei Textdaten klassifiziert es >95% der Artikel als 'Rauschen' (Label -1). 
            * Eine Suche müsste also den 'Rauschen'-Haufen durchsuchen (fast alle Artikel), was **keinen Performance-Gewinn** bringt.
            """)

        k_clusters = st.slider("Anzahl der Cluster (k) für die Vorsortierung", 5, 50, 15, key="search_k")
        
        if st.button("Vorbereitung & K-Means Clustering starten!", use_container_width=True, key="search_start_cluster"):
            with st.spinner(f"Führe K-Means (k={k_clusters}) aus..."):
                model, labels, cluster_time = run_kmeans_clustering(X_tfidf, k_clusters)
                st.session_state["search_model"] = model
                st.session_state["search_labels"] = labels
                st.session_state["search_algo_name"] = "K-Means"
            
            st.success(f"Clustering abgeschlossen in {cluster_time:.2f} Sekunden. {k_clusters} Cluster gefunden.")

        # --- Schritt 2: Cluster-Analyse ---
        if "search_labels" in st.session_state:
            st.divider()
            st.header("Schritt 2: Cluster-Analyse (optional)")
            
            labels = st.session_state["search_labels"]
            model = st.session_state["search_model"]
            
            unique_labels = sorted(list(set(labels)))
            display_labels = [f"Cluster {l} ({np.sum(labels == l)} Artikel)" for l in unique_labels]
            label_map = {display: value for display, value in zip(display_labels, unique_labels)}

            selected_display_label = st.selectbox("Wähle einen Cluster, um den Inhalt zu prüfen:", display_labels, key="search_cluster_select")
            selected_label = label_map[selected_display_label]
            
            st.subheader(f"Thema für: {selected_display_label}")
            centroid_vector = model.cluster_centers_[selected_label]
            terms = vectorizer.get_feature_names_out()
            top_indices = centroid_vector.argsort()[-5:][::-1]
            top_words = [terms[i] for i in top_indices]
            st.info(f"**Top 5 Wörter (Thema):** `{', '.join(top_words)}`")
            
            st.subheader(f"Beispiel-Artikel für: {selected_display_label}")
            cluster_indices = np.where(labels == selected_label)[0]
            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(cluster_indices, size=min(5, len(cluster_indices)), replace=False)
                for i in sample_indices:
                    with st.expander(f"Artikel {i}"):
                        st.write(texts[i])
            else:
                st.write("Keine Artikel in diesem Cluster.")

        st.divider()

        # --- 3. SUCH-BEREICH ---
        st.header("Schritt 3: Suche vergleichen")
        query = st.text_input("Gib deinen Suchbegriff ein (z.B. 'computer' oder 'sports'):", "computer", key="search_query")
        
        k_search = 1
        start_button = None
        
        if "search_model" in st.session_state:
            max_k = st.session_state["search_model"].n_clusters
            
            st.markdown(f"""
            **Empfehlung für "Top N":**
            Bei **{max_k}** Clustern in Schritt 1 ist ein "Top N"-Wert von **{max(1, int(max_k * 0.1))} bis {max(2, int(max_k * 0.2))}** (ca. 10-20%) oft ein guter Kompromiss zwischen Geschwindigkeit und Genauigkeit.
            """)
            
            col_search1, col_search2, col_search3 = st.columns([1.5, 1, 1.5]) 
            with col_search1:
                k_search = st.number_input(
                    "Wie viele Cluster durchsuchen? (Top N)",
                    min_value=1, 
                    max_value=max_k, 
                    value=1, 
                    step=1,
                    help="N=1 ist am schnellsten, aber ungenau. N erhöhen steigert die Genauigkeit (Recall), aber verlangsamt die Suche."
                ) 
            with col_search2:
                st.write("")
                start_button = st.button("Suche starten!", type="primary", key="search_start_search") 
        else:
            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
            with col_btn2:
                start_button = st.button("Suche starten!", type="primary", key="search_start_search_fallback")
        
        if start_button and query:
            if "search_model" not in st.session_state:
                st.error("Bitte führe zuerst 'Schritt 1: Vorbereitung & Clustering' aus.")
            else:
                st.header(f"Ergebnisse für: '{query}'")
                col1, col2 = st.columns(2)
                
                # --- METHODE 1: LINEARE SUCHE (BASELINE) ---
                with col1:
                    st.subheader("Lineare Suche (Baseline)")
                    start_time_linear = time.perf_counter()
                    with st.spinner(f"Durchsuche alle {len(texts)} Artikel..."):
                        matches_linear_indices_set = set(linear_search(texts, query))
                    end_time_linear = time.perf_counter()
                    time_linear_ms = (end_time_linear - start_time_linear) * 1000
                    count_linear = len(matches_linear_indices_set)
                    
                    st.metric("Ausführungszeit", f"{time_linear_ms:.2f} ms")
                    st.metric("Gefundene Treffer", f"{count_linear}")
                    st.metric("Genauigkeit (Recall)", "100 % (Baseline)")
                    st.info(f"**Was ist passiert?**\nEs wurden alle **{len(texts)}** Artikel nacheinander durchsucht.")
                    st.write(f"**Beispiel-Treffer (Top 3 von {count_linear}):**")
                    for i in list(matches_linear_indices_set)[:3]:
                        with st.expander(f"Artikel {i} (Auszug)"):
                            st.write(texts[i][:500] + "...")

                # --- METHODE 2: K-MEANS-BASIERTE SUCHE (HEURISTISCH) ---
                with col2:
                    st.subheader(f"K-Means-Suche (Top {k_search} Cluster)")
                    model = st.session_state["search_model"]
                    labels = st.session_state["search_labels"]
                    
                    start_time_cluster = time.perf_counter()
                    query_vector = vectorizer.transform([query])
                    distances = model.transform(query_vector)[0]
                    target_clusters = distances.argsort()[:k_search]
                    
                    final_indices_to_search = []
                    articles_to_search_count = 0
                    for cluster_id in target_clusters:
                        cluster_indices = np.where(labels == cluster_id)[0]
                        final_indices_to_search.extend(cluster_indices)
                        articles_to_search_count += len(cluster_indices)
                    
                    final_indices_to_search = list(set(final_indices_to_search))
                    texts_subset = [texts[i] for i in final_indices_to_search]
                    
                    subset_matches_indices = linear_search(texts_subset, query)
                    matches_cluster_indices_set = set(
                        [final_indices_to_search[i] for i in subset_matches_indices]
                    )
                    
                    end_time_cluster = time.perf_counter()
                    time_cluster_ms = (end_time_cluster - start_time_cluster) * 1000
                    count_cluster = len(matches_cluster_indices_set)

                    if count_linear > 0:
                        correct_hits = len(matches_cluster_indices_set.intersection(matches_linear_indices_set))
                        accuracy_percent = (correct_hits / count_linear) * 100
                    else: 
                        accuracy_percent = 100.0

                    st.metric("Ausführungszeit", f"{time_cluster_ms:.2f} ms")
                    st.metric("Gefundene Treffer", f"{count_cluster}")
                    st.metric("Genauigkeit (Recall)", f"{accuracy_percent:.1f} %")
                    st.info(f"**Was ist passiert?**\n1. Suche auf die **Top {k_search}** relevantesten Cluster beschränkt.\n2. Es wurden nur **{articles_to_search_count}** Artikel durchsucht.")
                    
                    st.write(f"**Beispiel-Treffer (Top 3 von {count_cluster}):**")
                    for i in list(matches_cluster_indices_set)[:3]:
                        with st.expander(f"Artikel {i} (Auszug)"):
                            st.write(texts[i][:500] + "...")
                
                # --- Fazit ---
                st.divider()
                st.header(f"Fazit für: '{query}'")
                if time_cluster_ms > 0 and time_linear_ms > 0:
                    speedup = time_linear_ms / time_cluster_ms
                    st.success(f"Die **K-Means-Suche** war **{speedup:.1f}x schneller** als die lineare Suche.")
                    if accuracy_percent < 95:
                        st.warning(f"**Genauigkeits-Kompromiss:** K-Means hat **{100 - accuracy_percent:.1f}%** der relevanten Treffer übersehen (Recall: {accuracy_percent:.1f}%).")
                    else:
                        st.info(f"**Hohe Genauigkeit:** Mit Top-{k_search} Clustern war K-Means sehr genau (Recall: {accuracy_percent:.1f}%) UND deutlich schneller.")

    except Exception as e:
        st.error(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
        st.exception(e)


# --- #################### TAB 2 #################### ---
with tab2:
    st.header("Algorithmen-Demonstration mit 2D-Punkten")
    st.info("""
    **Zweck:** Dieser Seite soll zeigen, wie die Algorithmen *prinzipiell* funktionieren. 
    Dafür werden 2D-Punkte statt Text genutzt, damit man die Ergebnisse *sehen* kann. 
    Dies hilft zu verstehen, warum K-Means auf Seite 1 funktioniert, DBSCAN und hierarchisches Clustering aber scheitern würden.
    """)

    demo_col1, demo_col2 = st.columns([1, 1])
    
    with demo_col1:
        st.subheader("Steuerung")
        
        dataset_name = st.selectbox(
            "Wähle einen Test-Datensatz:",
            ("Monde (Moons)", "Kreise (Circles)", "Blobs (Haufen)")
        )
        X_spiel = lade_spielplatz_daten(dataset_name)
        
        algo_name_spiel = st.selectbox(
            "Wähle einen Algorithmus:",
            ("K-Means", "DBSCAN", "Hierarchisches Clustering")
        )
        
        params_spiel = {}
        if algo_name_spiel == "K-Means":
            params_spiel["k"] = st.slider("Anzahl der Cluster (k)", 2, 5, 2, key="demo_k")
        elif algo_name_spiel == "DBSCAN":
            params_spiel["eps"] = st.slider("Epsilon (eps)", 0.1, 1.0, 0.5, 0.05, key="demo_eps")
            params_spiel["min_samples"] = st.slider("Minimale Samples", 2, 10, 5, key="demo_min_samples")
        elif algo_name_spiel == "Hierarchisches Clustering":
            params_spiel["n_clusters"] = st.slider("Anzahl der Cluster", 2, 5, 2, key="demo_n_clusters")
            params_spiel["linkage"] = st.selectbox("Linkage-Methode", ("ward", "complete", "average"), key="demo_linkage")
            if params_spiel["linkage"] != "ward":
                 params_spiel.pop("n_clusters")
                 params_spiel["distance_threshold"] = st.slider("Distanz-Schwelle", 0.1, 5.0, 1.5, 0.1, key="demo_dist")
        
        start_spielplatz = st.button("Clustering starten", key="demo_start")
        
        st.divider()
        st.subheader("Ergebnisse")
        if algo_name_spiel == "K-Means":
            st.markdown("**K-Means:** Versucht, 'runde' Cluster (wie Blobs) zu finden. Er weist jeden Punkt einem Cluster zu.")
            if dataset_name != "Blobs (Haufen)":
                st.warning(f"**Erkenntnis:** K-Means scheitert bei komplexen Formen wie '{dataset_name}', da dieser nur auf die 'Mittelpunkte' achtet.")
                
        elif algo_name_spiel == "DBSCAN":
            st.markdown("**DBSCAN:** Findet Cluster basierend auf 'Dichte'. Er kann komplexe Formen finden und **Rauschen** (graue 'x') identifizieren.")
            if dataset_name != "Blobs (Haufen)":
                st.success(f"**Erkenntnis 1:** DBSCAN ist hervorragend für komplexe 2D-Formen wie '{dataset_name}'.")
            st.warning("**Erkenntnis 2:** DBSCAN hat das Konzept von 'Rauschen' (Noise). Das ist der Grund, warum dieser auf Seite 1 für die Suche ungeeignet ist – dort sind 95% der Text-Daten 'Rauschen'.")
            
        elif algo_name_spiel == "Hierarchisches Clustering":
            st.markdown("**Hierarchisches Clustering:** Baut einen 'Stammbaum' der Änhlichkeit. Man kann dann 'schneiden', um eine bestimmte Anzahl von Clustern zu erhalten.")
            st.warning("**Erkenntnis:** Dieser Algorithmus ist sehr rechen- und speicherintensiv (skaliert quadratisch, O(n²)). Er ist für große Datensätze wie die 70.000 Artikel auf Seite 1 **rechnerisch unmöglich**.")

    
    with demo_col2:
        st.subheader("Ergebnis")
        
        if start_spielplatz:
            try:
                model_spiel = None
                if algo_name_spiel == "K-Means":
                    model_spiel = KMeans(n_clusters=params_spiel["k"], random_state=42, n_init=10)
                elif algo_name_spiel == "DBSCAN":
                    model_spiel = DBSCAN(eps=params_spiel["eps"], min_samples=params_spiel["min_samples"])
                elif algo_name_spiel == "Hierarchisches Clustering":
                    model_spiel = AgglomerativeClustering(
                        n_clusters=params_spiel.get("n_clusters"), 
                        linkage=params_spiel.get("linkage", "ward"),
                        distance_threshold=params_spiel.get("distance_threshold")
                    )
                
                start_t = time.perf_counter()
                labels_spiel = model_spiel.fit_predict(X_spiel)
                end_t = time.perf_counter()
                time_spiel_ms = (end_t - start_t) * 1000
                
                st.write(f"Clustering berechnet in: **{time_spiel_ms:.2f} ms**")
                fig_clustered = plotte_spielplatz(X_spiel, labels_spiel, algo_name_spiel)
                st.pyplot(fig_clustered)
                
                st.session_state["last_spiel_fig"] = fig_clustered
                st.session_state["last_spiel_time"] = time_spiel_ms
                st.session_state["last_spiel_algo"] = algo_name_spiel
            
            except Exception as e:
                st.error(f"Ein Fehler ist aufgetreten: {e}")
                st.info("Bitte überprüfe die Parameter-Kombination (z.B. 'ward' Linkage funktioniert nur mit 'Anzahl der Cluster', nicht 'Distanz-Schwelle').")
                st.session_state["last_spiel_fig"] = None 
        
        elif "last_spiel_fig" in st.session_state and st.session_state["last_spiel_fig"] is not None:
            st.write(f"Letztes Ergebnis ({st.session_state['last_spiel_algo']}) berechnet in: **{st.session_state['last_spiel_time']:.2f} ms**")
            st.pyplot(st.session_state["last_spiel_fig"])
        
        else:
            st.info("Bitte links Parameter auswählen und 'Clustering starten' klicken.")

# --- Versionsnummer am Ende der Seite ---
st.divider()
st.caption(APP_VERSION)