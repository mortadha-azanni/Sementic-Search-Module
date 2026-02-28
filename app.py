"""
Streamlit UI for the Bakery/Pastry Semantic Search Module.
Run with: uv run streamlit run app.py
"""

import streamlit as st
from core.embedding import Embedder
from core.database import VectorDB

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rose Blanche – Recherche Sémantique",
    page_icon="🥐",
    layout="centered",
)

# ── Cached singletons (loaded once, reused across reruns) ───
@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def load_embedder():
    return Embedder()


@st.cache_resource(show_spinner="Connexion à la base de données…")
def load_db():
    db = VectorDB()
    if not db.conn:
        return None
    return db


def get_db():
    """Get DB connection, reconnecting if it was dropped."""
    db = load_db()
    if db is None:
        return None
    # Check if connection is still alive
    try:
        db.conn.cursor().execute("SELECT 1")
    except Exception:
        # Connection died — clear cache and reconnect
        load_db.clear()
        db = load_db()
    return db


# ── Init ─────────────────────────────────────────────────────
embedder = load_embedder()
db = get_db()

# ── Header ───────────────────────────────────────────────────
st.title("🥐 Rose Blanche — Recherche Sémantique")
st.caption(
    "Module de recherche sémantique pour la formulation en boulangerie & pâtisserie. "
    "Posez une question en langage naturel pour retrouver les passages les plus pertinents."
)

# ── DB status ────────────────────────────────────────────────
if db is None:
    st.error("❌ Impossible de se connecter à la base de données. Vérifiez votre fichier `.env`.")
    st.stop()

row_count = db.count_embeddings()
if row_count == 0:
    st.warning("⚠️ La base est vide. Exécutez `uv run ingest.py` d'abord.")
    st.stop()

st.info(f"📊 Base de données : **{row_count}** fragments indexés depuis **35** fiches techniques.")

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")
    top_k = st.slider("Nombre de résultats (Top K)", min_value=1, max_value=20, value=3)
    min_score = st.slider("Score minimum", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    st.divider()
    st.markdown(
        "**Tech stack**\n"
        "- 🧠 all-MiniLM-L6-v2 (384d)\n"
        "- 🐘 PostgreSQL 14 + pgvector\n"
        "- 📐 Cosine similarity\n"
    )

# ── Search form ──────────────────────────────────────────────
question = st.text_input(
    "🔍 Votre question",
    placeholder="Ex : Quel est le dosage recommandé pour l'acide ascorbique ?",
)

search_clicked = st.button("Rechercher", type="primary", use_container_width=True)

# ── Results ──────────────────────────────────────────────────
if search_clicked and question.strip():
    with st.spinner("Recherche en cours…"):
        query_vector = embedder.encode_query(question.strip())
        results = db.search_similar_fragments(query_vector, top_k=top_k)

    # Filter by minimum score
    results = [r for r in results if float(r["similarity"]) >= min_score]

    if not results:
        st.warning("Aucun résultat trouvé.")
    else:
        st.success(f"**{len(results)}** résultat(s) trouvé(s)")
        st.divider()

        for i, res in enumerate(results, 1):
            score = float(res["similarity"])
            doc_id = res["id_document"]

            # Color-code the score
            if score >= 0.6:
                score_color = "🟢"
            elif score >= 0.4:
                score_color = "🟡"
            else:
                score_color = "🔴"

            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### Résultat {i}")
                with col2:
                    st.metric("Score", f"{score:.2f}", label_visibility="collapsed")

                st.markdown(f"**Document** : fiche n°{doc_id}")
                st.info(f'"{res["texte_fragment"]}"')
                st.caption(f"{score_color} Similarité cosinus : **{score:.4f}**")
                st.divider()

elif search_clicked:
    st.warning("Veuillez entrer une question.")
