"""NotebookRAGEngine — Frozen retrieval + generation pipeline.

Extracted from NB03 (Query Engine) + NB05 (Query Expansion & Reranking).
Hybrid FAISS+BM25 with normalization, ES<->EN expansion, and multi-signal reranking.

Usage:
    from src.rag.query_engine import NotebookRAGEngine
    engine = NotebookRAGEngine()
    engine.search("PCA")
    engine.ask("Que es Bag of Words?", mode="explicar")
    engine.summarize_session_part("llm_39", part="second_half")
"""

import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars or constructor args)
# ---------------------------------------------------------------------------
_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-minilm-l12-v2",
)
_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "")
_LOCAL_ONLY = os.environ.get("EMBEDDING_LOCAL_ONLY", "1").lower() in ("1", "true", "yes")
_ALLOW_DOWNLOAD = os.environ.get("EMBEDDING_ALLOW_DOWNLOAD", "0").lower() in ("1", "true", "yes")
_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_MODEL = "qwen2.5:7b"


def _load_embedding_model(
    model_name: str = _MODEL_NAME,
    model_path: Optional[str] = _MODEL_PATH or None,
    local_files_only: bool = _LOCAL_ONLY,
    allow_download: bool = _ALLOW_DOWNLOAD,
) -> SentenceTransformer:
    """Load SentenceTransformer with offline-first strategy.

    Order:
      1. Explicit local path (model_path)
      2. HF cache with local_files_only=True
      3. Online download (only if allow_download=True)
    """
    # 1. Explicit local path
    if model_path:
        p = Path(model_path)
        if p.is_dir():
            print(f"Loading embeddings model from local path: {p}")
            return SentenceTransformer(str(p))
        raise FileNotFoundError(
            f"embedding_model_path does not exist: {p}\n"
            "Set EMBEDDING_MODEL_PATH to a valid directory or remove it to use cache."
        )

    # 2. Local cache only
    if local_files_only:
        print(f"Loading embeddings model from local cache only: {model_name}")
        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception as cache_err:
            if not allow_download:
                raise RuntimeError(
                    f"Model '{model_name}' not found in local cache. Online download disabled.\n"
                    f"Options:\n"
                    f"  1. Set EMBEDDING_MODEL_PATH to a local model directory\n"
                    f"  2. Set EMBEDDING_ALLOW_DOWNLOAD=1 to allow one-time download\n"
                    f"  3. Manually download: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\"\n"
                    f"Cache error: {cache_err}"
                ) from cache_err
            print(f"Not in cache. Falling through to online download...")

    # 3. Online download
    if allow_download:
        print(f"Downloading embeddings model: {model_name}")
        return SentenceTransformer(model_name)

    raise RuntimeError(
        f"Model '{model_name}' not available and online download disabled.\n"
        "Set EMBEDDING_ALLOW_DOWNLOAD=1 or EMBEDDING_MODEL_PATH to a local path."
    )

# ---------------------------------------------------------------------------
# BM25 Index (loaded from pickle built in NB02)
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25 index deserialized from pickle."""

    def __init__(self):
        pass

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        for k, v in data.items():
            setattr(idx, k, v)
        return idx

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-záéíóúüñ0-9_-]{2,}\b", text.lower())

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        scores = []
        for i in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lens[i]
            tf_dict = self.term_freqs[i]
            for term in query_tokens:
                if term not in tf_dict:
                    continue
                tf = tf_dict[term]
                df = self.doc_freqs.get(term, 0)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )
                score += idf * tf_norm
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "bm25_score": round(score, 4),
                "page_content": doc["page_content"],
                "topic": doc.get("topic", ""),
                "keywords": doc.get("keywords", []),
                "session_id": doc.get("session_id", ""),
                "start_ms": doc.get("start_ms", 0),
                "end_ms": doc.get("end_ms", 0),
                "quality_score": doc.get("quality_score", 0),
                "linked_doc_id": doc.get("linked_doc_id", ""),
            })
        return results


# ---------------------------------------------------------------------------
# Query normalization (NB05)
# ---------------------------------------------------------------------------

QUERY_STOPWORDS = {
    "que", "qué", "como", "cómo", "cual", "cuál", "cuales", "cuáles",
    "cuando", "cuándo", "donde", "dónde", "quien", "quién",
    "es", "son", "fue", "era", "ser", "estar", "hay", "tiene",
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "con", "por", "para",
    "me", "se", "te", "le", "lo", "nos",
    "y", "o", "pero", "si", "no",
    "explicame", "explícame", "dime", "dame", "muestrame", "muéstrame",
    "puedes", "podrias", "podrías", "favor", "por favor",
    "hablame", "háblame", "cuentame", "cuéntame",
}

INTENT_PATTERNS = [
    (r"\bdiferencia(?:s)?\b.*\bentre\b", "comparacion"),
    (r"\bvs\b|\bversus\b|\bfrente a\b", "comparacion"),
    (r"\bque es\b|\bqué es\b|\bdefinicion\b|\bdefinición\b", "definicion"),
    (r"\bresume\b|\bresumen\b|\bresumeme\b|\bresúmeme\b", "resumen"),
    (r"\bexamen\b|\bpregunta(?:s)?\b|\bejercicio(?:s)?\b", "examen"),
    (r"\bejemplo(?:s)?\b|\bcaso(?:s)? practico\b", "ejemplo"),
    (r"\bcomo funciona\b|\bcómo funciona\b|\bpara que sirve\b", "explicacion"),
]


@dataclass
class NormalizedQuery:
    original: str
    normalized: str
    core_terms: List[str]
    intent: str
    entities: List[str]


# ---------------------------------------------------------------------------
# Prompt templates (NB03)
# ---------------------------------------------------------------------------

_REGLAS_BASE = """REGLAS ESTRICTAS:
1. Responde UNICAMENTE con informacion que aparezca en los fragmentos. NO completes con tu conocimiento.
2. Estructura tu respuesta en dos bloques:
   **Segun los apuntes:** (solo lo que dicen los fragmentos, citando [Fuente N, MM:SS])
   **Nota adicional:** (solo si es imprescindible para entender, marcado como "esto no aparece en el material")
3. Si los fragmentos no contienen la respuesta, di: "Los fragmentos recuperados no cubren este tema directamente. Lo que se menciona es: ..." y describe lo que SI aparece.
4. NUNCA presentes conocimiento general como si viniera de los apuntes.
5. Usa siempre: "el profesor menciona", "en la clase se explica", "segun el fragmento". NUNCA afirmes en absoluto."""

SYSTEM_PROMPTS = {
    "explicar": f"""Tutor que explica SOLO lo que dicen los apuntes de clase recuperados.
NO completes con conocimiento propio. Si el fragmento habla de "gradiente" pero la pregunta es sobre "red neuronal", di que los apuntes no cubren ese tema directamente y describe lo que SI aparece.
{_REGLAS_BASE}""",

    "tecnico": f"""Experto tecnico. Responde con precision usando SOLO los fragmentos.
Si necesitas añadir una definicion tecnica no presente en los apuntes, separala claramente con "Nota: esto no aparece en el material de clase".
{_REGLAS_BASE}""",

    "examen": f"""Profesor. Genera 3 preguntas de comprension y 2 de aplicacion.
Las preguntas y respuestas deben poder responderse SOLO con los fragmentos. No incluyas preguntas cuya respuesta requiera conocimiento externo.
{_REGLAS_BASE}""",

    "resumen": f"""Genera un resumen de lo que dicen los fragmentos. Cada punto debe citar su fuente.
No generalices ni añadas conclusiones que no esten explicitas en el material.
{_REGLAS_BASE}""",
}

GENERIC_TOPICS = {
    "segmento de clase", "clase", "tema", "concepto", "explicacion",
    "contenido", "material", "sesion", "introduccion",
}

GENERIC_EXPLANATIONS = {
    "consulta la grabación", "ver la grabación", "ver el vídeo",
    "revisa el material", "segmento de clase",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timestamp(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    return f"{minutes}:{seconds:02d}"


def _build_section(chunks: List[Dict]) -> Dict:
    """Build a section dict from a list of chunks, picking top 2 topic labels."""
    from collections import Counter
    topics = [
        (c.get("topic", "") or "").strip()
        for c in chunks
        if (c.get("topic", "") or "").strip().lower() not in GENERIC_TOPICS
    ]
    topic_counts = Counter(topics)
    # Pick up to 2 most common non-generic topics
    top_topics = [t for t, _ in topic_counts.most_common(3) if t][:2]
    label = " / ".join(top_topics) if top_topics else "Contenido general"
    return {
        "topic": label,
        "chunks": chunks,
        "start_ms": chunks[0].get("start_ms", 0),
        "end_ms": chunks[-1].get("end_ms", 0),
    }


# ---------------------------------------------------------------------------
# NotebookRAGEngine
# ---------------------------------------------------------------------------

class NotebookRAGEngine:
    """Frozen RAG engine from NB03+NB05: hybrid FAISS+BM25 with expansion & reranking.

    Does NOT touch or conflict with the existing RAGEngine in engine.py.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        ollama_url: str = _OLLAMA_URL,
        ollama_model: str = _OLLAMA_MODEL,
        embedding_model_name: str = _MODEL_NAME,
        embedding_model_path: Optional[str] = _MODEL_PATH or None,
        local_files_only: bool = _LOCAL_ONLY,
        allow_download: bool = _ALLOW_DOWNLOAD,
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent
        self.project_root = project_root
        self.index_dir = project_root / ".rag_index"
        self.rag_dir = project_root / "artifacts" / "rag"
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Embedding model (offline-first)
        self.embed_model = _load_embedding_model(
            model_name=embedding_model_name,
            model_path=embedding_model_path,
            local_files_only=local_files_only,
            allow_download=allow_download,
        )

        # FAISS indices
        self.faiss_enriched, self.meta_enriched = self._load_faiss("pseudo_label")
        self.faiss_evidence, self.meta_evidence = self._load_faiss("raw_evidence")

        # BM25 indices
        self.bm25_enriched = BM25Index.load(self.index_dir / "bm25_pseudo_label.pkl")

        # Expansion dictionaries (from NB05 → expansion_config.json)
        self.synonym_map, self.acronym_map = self._load_expansion_config()

        # Known topics for entity detection
        self._known_topics = {
            m.get("topic", "").lower().strip()
            for m in self.meta_enriched if m.get("topic")
        }

        # Session index: session_id → list of metadata dicts, sorted by start_ms
        self._session_index = self._build_session_index()

        # Topic index: session_id → ranked list of topic summaries
        self._topic_index = self._build_topic_index()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_faiss(self, name: str) -> Tuple[faiss.Index, List[Dict]]:
        cdir = self.index_dir / name
        index = faiss.read_index(str(cdir / "index.faiss"))
        with open(cdir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    def _load_expansion_config(self) -> Tuple[Dict, Dict]:
        config_path = self.rag_dir / "expansion_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            return cfg.get("synonym_map", {}), cfg.get("acronym_map", {})
        return {}, {}

    def _build_session_index(self) -> Dict[str, List[Dict]]:
        sessions: Dict[str, List[Dict]] = {}
        for m in self.meta_enriched:
            sid = m.get("session_id", "")
            if sid:
                sessions.setdefault(sid, []).append(m)
        for sid in sessions:
            sessions[sid].sort(key=lambda x: x.get("start_ms", 0))
        return sessions

    def _build_topic_index(self) -> Dict[str, List[Dict]]:
        """Group topics by session with count, timestamps, and quality."""
        from collections import defaultdict

        raw: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        for m in self.meta_enriched:
            sid = m.get("session_id", "")
            topic = (m.get("topic", "") or "").strip()
            if not sid or not topic or topic.lower() in GENERIC_TOPICS:
                continue
            key = topic.lower()
            if key not in raw[sid]:
                raw[sid][key] = {
                    "topic": topic,
                    "count": 0,
                    "first_ms": m.get("start_ms", 0),
                    "last_ms": m.get("end_ms", 0),
                    "quality_sum": 0.0,
                }
            entry = raw[sid][key]
            entry["count"] += 1
            entry["first_ms"] = min(entry["first_ms"], m.get("start_ms", 0))
            entry["last_ms"] = max(entry["last_ms"], m.get("end_ms", 0))
            entry["quality_sum"] += m.get("quality_score", 0)

        result: Dict[str, List[Dict]] = {}
        for sid, topics in raw.items():
            ranked = []
            for entry in topics.values():
                ranked.append({
                    "topic": entry["topic"],
                    "count": entry["count"],
                    "first_ms": entry["first_ms"],
                    "last_ms": entry["last_ms"],
                    "avg_quality": round(entry["quality_sum"] / entry["count"], 3),
                })
            ranked.sort(key=lambda x: (x["count"], x["avg_quality"]), reverse=True)
            result[sid] = ranked
        return result

    def get_session_topics(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Return ranked list of topics for a session."""
        return self._topic_index.get(session_id, [])[:limit]

    # ------------------------------------------------------------------
    # Query normalization & expansion (NB05)
    # ------------------------------------------------------------------

    def normalize_query(self, query: str) -> NormalizedQuery:
        original = query.strip()
        intent = "general"
        for pattern, intent_type in INTENT_PATTERNS:
            if re.search(pattern, original, re.IGNORECASE):
                intent = intent_type
                break
        cleaned = original.lower()
        cleaned = re.sub(r"[¿¡?!.,;:()\[\]\"']", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = cleaned.split()
        core_terms = [t for t in tokens if t not in QUERY_STOPWORDS and len(t) > 1]
        normalized = " ".join(core_terms) if core_terms else cleaned
        entities = []
        for topic in self._known_topics:
            if topic and len(topic) > 2:
                topic_words = set(topic.split())
                query_words = set(cleaned.split())
                overlap = topic_words & query_words
                if overlap and len(overlap) / len(topic_words) >= 0.5:
                    entities.append(topic)
        return NormalizedQuery(
            original=original, normalized=normalized,
            core_terms=core_terms, intent=intent,
            entities=sorted(set(entities)),
        )

    def expand_query(self, nq: NormalizedQuery) -> List[str]:
        expanded_terms = set(nq.core_terms)
        text_lower = nq.normalized.lower()
        for term, synonyms in self.synonym_map.items():
            if term in text_lower:
                for syn in synonyms:
                    expanded_terms.update(syn.lower().split())
        for term in nq.core_terms:
            tl = term.lower()
            if tl in self.acronym_map:
                expanded_terms.update(self.acronym_map[tl].lower().split())
        for entity in nq.entities:
            expanded_terms.update(entity.lower().split())
        queries = [nq.normalized]
        expanded_text = " ".join(sorted(expanded_terms))
        if expanded_text != nq.normalized:
            queries.append(expanded_text)
        for entity in nq.entities[:2]:
            if entity.lower() != nq.normalized.lower():
                queries.append(entity)
        return queries

    # ------------------------------------------------------------------
    # FAISS search
    # ------------------------------------------------------------------

    def _search_faiss(
        self, query: str, index: faiss.Index, metadata: List[Dict], top_k: int = 5,
    ) -> List[Dict]:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb)
        scores, indices = index.search(q_emb.astype(np.float32), min(top_k, index.ntotal))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            meta = metadata[int(idx)]
            results.append({
                "doc_id": meta["doc_id"],
                "similarity": float(scores[0][i]),
                "page_content": meta["page_content"],
                "topic": meta.get("topic", ""),
                "keywords": meta.get("keywords", []),
                "session_id": meta.get("session_id", ""),
                "start_ms": meta.get("start_ms", 0),
                "end_ms": meta.get("end_ms", 0),
                "quality_score": meta.get("quality_score", 0),
                "linked_doc_id": meta.get("linked_doc_id", ""),
                "source": "faiss",
            })
        return results

    # ------------------------------------------------------------------
    # Multi-query retrieval (NB05)
    # ------------------------------------------------------------------

    def _multi_query_retrieval(self, queries: List[str], top_k_per_query: int = 10) -> List[Dict]:
        doc_pool: Dict[str, Dict] = {}
        for query in queries:
            faiss_results = self._search_faiss(
                query, self.faiss_enriched, self.meta_enriched, top_k=top_k_per_query,
            )
            for r in faiss_results:
                doc_id = r["doc_id"]
                if doc_id not in doc_pool:
                    doc_pool[doc_id] = {
                        **r, "faiss_score": r["similarity"],
                        "bm25_score": 0.0, "matched_queries": set(),
                    }
                else:
                    doc_pool[doc_id]["faiss_score"] = max(
                        doc_pool[doc_id].get("faiss_score", 0), r["similarity"],
                    )
                doc_pool[doc_id]["matched_queries"].add(query)

            bm25_results = self.bm25_enriched.search(query, top_k=top_k_per_query)
            max_bm25 = max((r["bm25_score"] for r in bm25_results), default=1.0) or 1.0
            for r in bm25_results:
                doc_id = r["doc_id"]
                bm25_norm = r["bm25_score"] / max_bm25
                if doc_id not in doc_pool:
                    doc_pool[doc_id] = {
                        **r, "faiss_score": 0.0, "bm25_score": bm25_norm,
                        "similarity": 0.0, "source": "bm25", "matched_queries": set(),
                    }
                else:
                    doc_pool[doc_id]["bm25_score"] = max(
                        doc_pool[doc_id].get("bm25_score", 0), bm25_norm,
                    )
                doc_pool[doc_id]["matched_queries"].add(query)

        for doc in doc_pool.values():
            doc["query_match_count"] = len(doc["matched_queries"])
            del doc["matched_queries"]
        return list(doc_pool.values())

    # ------------------------------------------------------------------
    # Reranking (NB05)
    # ------------------------------------------------------------------

    def _rerank(
        self,
        candidates: List[Dict],
        nq: NormalizedQuery,
        top_k: int = 5,
        w_faiss: float = 0.35,
        w_bm25: float = 0.25,
        w_topic: float = 0.15,
        w_keyword: float = 0.10,
        w_quality: float = 0.10,
        w_multi: float = 0.05,
        penalty_generic: float = 0.10,
        dedup_overlap: float = 0.8,
    ) -> List[Dict]:
        query_terms = set(nq.normalized.lower().split())
        max_qm = max((c.get("query_match_count", 1) for c in candidates), default=1)

        for doc in candidates:
            score = 0.0
            bd: Dict[str, float] = {}

            bd["faiss"] = doc.get("faiss_score", 0) * w_faiss
            score += bd["faiss"]

            bd["bm25"] = doc.get("bm25_score", 0) * w_bm25
            score += bd["bm25"]

            # Topic boost
            topic = doc.get("topic", "").lower()
            topic_words = set(topic.split())
            topic_overlap = len(query_terms & topic_words)
            topic_score = min(topic_overlap / max(len(topic_words), 1), 1.0)
            for entity in nq.entities:
                if entity.lower() in topic or topic in entity.lower():
                    topic_score = 1.0
                    break
            bd["topic"] = topic_score * w_topic
            score += bd["topic"]

            # Keyword boost
            doc_keywords = set(kw.lower() for kw in doc.get("keywords", []))
            kw_overlap = len(query_terms & doc_keywords)
            expanded_kw: Set[str] = set()
            for term in query_terms:
                if term in self.synonym_map:
                    for syn in self.synonym_map[term]:
                        expanded_kw.update(syn.lower().split())
                if term in self.acronym_map:
                    expanded_kw.update(self.acronym_map[term].lower().split())
            kw_overlap += len(expanded_kw & doc_keywords)
            kw_score = min(kw_overlap / max(len(doc_keywords), 1), 1.0)
            bd["keyword"] = kw_score * w_keyword
            score += bd["keyword"]

            bd["quality"] = doc.get("quality_score", 0) * w_quality
            score += bd["quality"]

            bd["multi"] = (doc.get("query_match_count", 1) / max(max_qm, 1)) * w_multi
            score += bd["multi"]

            # Generic penalty
            penalty = 0.0
            if topic in GENERIC_TOPICS:
                penalty += 0.5
            content = doc.get("page_content", "").lower()
            if any(g in content for g in GENERIC_EXPLANATIONS):
                penalty += 0.5
            if len(content) < 100:
                penalty += 0.3
            bd["penalty"] = -min(penalty, 1.0) * penalty_generic
            score += bd["penalty"]

            doc["rerank_score"] = round(max(score, 0), 4)
            doc["rerank_breakdown"] = bd

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Dedup by temporal overlap
        final = []
        for doc in candidates:
            dominated = False
            for sel in final:
                s1, e1 = doc.get("start_ms", 0), doc.get("end_ms", 0)
                s2, e2 = sel.get("start_ms", 0), sel.get("end_ms", 0)
                if s1 and e1 and s2 and e2:
                    ov = max(0, min(e1, e2) - max(s1, s2))
                    if ov / max(e1 - s1, 1) > dedup_overlap:
                        dominated = True
                        break
            if not dominated:
                final.append(doc)
            if len(final) >= top_k:
                break
        return final

    # ------------------------------------------------------------------
    # Evidence lookup
    # ------------------------------------------------------------------

    def _get_evidence(self, doc_id: str) -> Optional[Dict]:
        linked_id = None
        for meta in self.meta_enriched:
            if meta["doc_id"] == doc_id:
                linked_id = meta.get("linked_doc_id", "")
                break
        if linked_id:
            for meta in self.meta_evidence:
                if meta["doc_id"] == linked_id:
                    return meta
        return None

    # ------------------------------------------------------------------
    # Context building (NB03)
    # ------------------------------------------------------------------

    def build_context_from_chunks(
        self,
        chunks: List[Dict],
        include_evidence: bool = True,
        max_chars: int = 4000,
    ) -> str:
        """Build LLM context string from a list of chunk dicts."""
        parts = []
        total = 0
        for i, r in enumerate(chunks, 1):
            start_fmt = format_timestamp(r.get("start_ms", 0))
            end_fmt = format_timestamp(r.get("end_ms", 0))
            quality = r.get("quality_score", 0)
            section = (
                f"--- Fuente {i} (calidad: {quality:.2f}, tiempo: {start_fmt}-{end_fmt}) ---\n"
                f"{r.get('page_content', '')}"
            )
            if include_evidence:
                ev = self._get_evidence(r["doc_id"])
                if ev:
                    section += f"\n\n[Evidencia original]:\n{ev.get('page_content', '')[:500]}"
            if total + len(section) > max_chars:
                break
            parts.append(section)
            total += len(section)
        return "\n\n".join(parts)

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        return f"""FRAGMENTOS RECUPERADOS DE CLASES GRABADAS:
{context}

PREGUNTA DEL ALUMNO: {query}

INSTRUCCIONES:
- Responde SOLO con lo que aparece en los fragmentos de arriba.
- Si los fragmentos no cubren la pregunta, dilo y describe lo que SI mencionan.
- Formato obligatorio:
  **Segun los apuntes:** ...cita cada punto con [Fuente N, MM:SS]...
  **Nota adicional:** ...(solo si es imprescindible, marcado como conocimiento externo)...
- Si no necesitas nota adicional, omite esa seccion."""

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def get_session_ids(self) -> List[str]:
        """Return available session IDs."""
        return sorted(self._session_index.keys())

    def get_session_chunks(self, session_id: str) -> List[Dict]:
        """Return all enriched chunks for a session, sorted by start_ms."""
        return list(self._session_index.get(session_id, []))

    def compute_session_range(
        self,
        session_id: str,
        part: str = "all",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Compute (start_ms, end_ms) for a session part.

        Args:
            session_id: e.g. "llm_39"
            part: "all", "first_half", "second_half", "first_third",
                  "middle_third", "last_third", "custom"
            start_ms: override start (only with part="custom")
            end_ms: override end (only with part="custom")
        """
        chunks = self.get_session_chunks(session_id)
        if not chunks:
            raise ValueError(f"Session '{session_id}' not found. Available: {self.get_session_ids()}")

        sess_start = chunks[0]["start_ms"]
        sess_end = chunks[-1]["end_ms"]
        duration = sess_end - sess_start

        if part == "custom":
            return (start_ms or sess_start, end_ms or sess_end)
        if part == "all":
            return (sess_start, sess_end)
        if part == "first_half":
            return (sess_start, sess_start + duration // 2)
        if part == "second_half":
            return (sess_start + duration // 2, sess_end)
        if part == "first_third":
            return (sess_start, sess_start + duration // 3)
        if part == "middle_third":
            return (sess_start + duration // 3, sess_start + 2 * duration // 3)
        if part == "last_third":
            return (sess_start + 2 * duration // 3, sess_end)
        raise ValueError(f"Unknown part '{part}'. Use: all, first_half, second_half, first_third, middle_third, last_third, custom")

    def _filter_chunks_by_range(
        self, chunks: List[Dict], start_ms: int, end_ms: int,
    ) -> List[Dict]:
        """Return chunks overlapping with [start_ms, end_ms]."""
        return [
            c for c in chunks
            if c.get("end_ms", 0) > start_ms and c.get("start_ms", 0) < end_ms
        ]

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    def check_ollama(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.ollama_model in m for m in models)
        except Exception:
            return False

    def _call_ollama(self, system: str, user_prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 800},
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat", json=payload, timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            return f"Error al generar respuesta con Ollama: {e}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Full enhanced search: normalize -> expand -> multi-query -> rerank.

        Returns list of ranked result dicts with rerank_score, topic, timestamps, etc.
        """
        nq = self.normalize_query(query)
        expanded = self.expand_query(nq)
        candidates = self._multi_query_retrieval(expanded, top_k_per_query=10)
        return self._rerank(candidates, nq, top_k=top_k)

    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search without LLM — returns formatted source summaries."""
        results = self.search(query, top_k=top_k)
        sources = []
        for r in results:
            sources.append({
                "topic": r.get("topic", "?"),
                "session_id": r.get("session_id", ""),
                "timestamp": f"{format_timestamp(r.get('start_ms', 0))} - {format_timestamp(r.get('end_ms', 0))}",
                "quality": round(r.get("quality_score", 0), 2),
                "score": round(r.get("rerank_score", 0), 4),
                "content_preview": r.get("page_content", "")[:200],
                "keywords": r.get("keywords", []),
            })
        return sources

    def ask(self, query: str, mode: str = "explicar", top_k: int = 5) -> Dict:
        """Full pipeline: search -> context -> Ollama -> response with sources."""
        t0 = time.time()

        # Retrieval
        t_ret = time.time()
        results = self.search(query, top_k=top_k)
        retrieval_time = time.time() - t_ret

        if not results:
            return {
                "response": "No encontre informacion relevante sobre este tema en las clases.",
                "sources": [],
                "timing": {"total_s": round(time.time() - t0, 2)},
            }

        # Context
        context = self.build_context_from_chunks(results)

        # LLM
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["explicar"])
        user_prompt = self._build_user_prompt(query, context)

        t_llm = time.time()
        response_text = self._call_ollama(system, user_prompt)
        llm_time = time.time() - t_llm

        # Sources
        sources = []
        for r in results:
            sources.append({
                "topic": r.get("topic", "?"),
                "session_id": r.get("session_id", ""),
                "timestamp": f"{format_timestamp(r.get('start_ms', 0))} - {format_timestamp(r.get('end_ms', 0))}",
                "quality": round(r.get("quality_score", 0), 2),
                "score": round(r.get("rerank_score", 0), 4),
            })

        return {
            "query": query,
            "mode": mode,
            "response": response_text,
            "sources": sources,
            "timing": {
                "retrieval_s": round(retrieval_time, 2),
                "llm_s": round(llm_time, 2),
                "total_s": round(time.time() - t0, 2),
            },
        }

    def cross_session_search(self, query: str, top_k: int = 20) -> Dict:
        """Search across all sessions and group results by session_id.

        Returns dict with sessions list (ranked by hit count) and total_hits.
        No LLM needed — pure retrieval.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return {"sessions": [], "total_hits": 0}

        from collections import defaultdict
        groups: Dict[str, Dict] = defaultdict(lambda: {
            "hits": [], "score_sum": 0.0, "topics": set(),
        })
        for r in results:
            sid = r.get("session_id", "unknown")
            g = groups[sid]
            g["hits"].append(r)
            g["score_sum"] += r.get("rerank_score", 0)
            topic = r.get("topic", "")
            if topic and topic.lower() not in GENERIC_TOPICS:
                g["topics"].add(topic)

        sessions = []
        for sid, g in groups.items():
            hits = g["hits"]
            starts = [h.get("start_ms", 0) for h in hits]
            ends = [h.get("end_ms", 0) for h in hits]
            sessions.append({
                "session_id": sid,
                "count": len(hits),
                "avg_score": round(g["score_sum"] / len(hits), 4),
                "topics": sorted(g["topics"]),
                "time_range": f"{format_timestamp(min(starts))} - {format_timestamp(max(ends))}",
            })
        sessions.sort(key=lambda x: (x["count"], x["avg_score"]), reverse=True)

        return {"sessions": sessions, "total_hits": len(results)}

    def diagnose(self, query: str, top_k: int = 5, mode: str = "explicar") -> str:
        """Diagnostic print: compares retrieval vs LLM response for a query.

        Shows chunks recovered, baseline (extractive), LLM response, and
        a faithfulness check to detect when the LLM adds unsupported info.
        """
        import textwrap

        out = []
        out.append("=" * 70)
        out.append(f"DIAGNOSTICO: {query}")
        out.append("=" * 70)

        # 1. Retrieval
        results = self.search(query, top_k=top_k)

        out.append(f"\n--- CHUNKS RECUPERADOS ({len(results)}) ---")
        if not results:
            out.append("  (ninguno)")
            return "\n".join(out)

        chunk_texts = []
        for i, r in enumerate(results, 1):
            topic = r.get("topic", "?")
            score = r.get("rerank_score", 0)
            start = format_timestamp(r.get("start_ms", 0))
            end = format_timestamp(r.get("end_ms", 0))
            session = r.get("session_id", "?")
            content = r.get("page_content", "")
            quality = r.get("quality_score", 0)

            out.append(f"\n  Fuente {i}: {topic}")
            out.append(f"    sesion={session} | {start}-{end} | score={score:.4f} | quality={quality:.2f}")
            # Show first 200 chars of content
            preview = content[:200].replace("\n", " ")
            out.append(f"    contenido: {preview}...")
            chunk_texts.append(content)

            # Show evidence if available
            ev = self._get_evidence(r["doc_id"])
            if ev:
                ev_preview = ev.get("page_content", "")[:150].replace("\n", " ")
                out.append(f"    evidencia: {ev_preview}...")

        # 2. Baseline extractive (no LLM)
        out.append(f"\n--- BASELINE SIN LLM (extractivo) ---")
        keywords_query = set(query.lower().split())
        relevant_sentences = []
        for text in chunk_texts:
            for sentence in re.split(r'[.!?\n]', text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                words = set(sentence.lower().split())
                overlap = len(keywords_query & words)
                if overlap >= 1:
                    relevant_sentences.append((overlap, sentence))

        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        if relevant_sentences:
            for _, sent in relevant_sentences[:5]:
                out.append(f"  - {sent[:120]}")
        else:
            out.append("  (sin coincidencias directas con la query)")

        # 3. LLM response
        context = self.build_context_from_chunks(results)
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["explicar"])
        user_prompt = self._build_user_prompt(query, context)
        response = self._call_ollama(system, user_prompt)

        out.append(f"\n--- RESPUESTA LLM ({mode}) ---")
        for line in response.split("\n"):
            out.append(f"  {line}")

        # 4. Faithfulness check
        out.append(f"\n--- CHEQUEO DE FIDELIDAD ---")
        # Check if response sentences appear grounded in chunks
        all_chunk_text = " ".join(chunk_texts).lower()
        response_sentences = [s.strip() for s in re.split(r'[.!?\n]', response) if len(s.strip()) > 20]

        # Disclaimer phrases — these are meta-commentary, not claims to check
        _DISCLAIMER_PATTERNS = [
            "no cubren este tema", "no aparece en el material", "no se menciona",
            "conocimiento externo", "nota adicional", "no cubre la pregunta",
            "fragmentos recuperados no", "esto no aparece",
        ]

        grounded = 0
        ungrounded = 0
        for sent in response_sentences:
            # Skip disclaimer/meta sentences — they're honest, not claims
            if any(d in sent.lower() for d in _DISCLAIMER_PATTERNS):
                continue

            # Extract key content words (>4 chars, not stopwords)
            content_words = [
                w for w in re.findall(r'\b\w{4,}\b', sent.lower())
                if w not in {"según", "segun", "clase", "fuente", "nota", "apuntes",
                             "fragmento", "profesor", "menciona", "explica", "adicional",
                             "aparece", "material", "general", "ejemplo", "indica",
                             "recuperados", "directamente", "cubren"}
            ]
            if not content_words:
                continue
            # How many content words appear in chunks?
            found = sum(1 for w in content_words if w in all_chunk_text)
            ratio = found / len(content_words) if content_words else 0

            if ratio >= 0.5:
                grounded += 1
            else:
                ungrounded += 1
                out.append(f"  [NO SOPORTADO] {sent[:100]}...")
                missing = [w for w in content_words if w not in all_chunk_text]
                out.append(f"    terminos no en chunks: {missing[:5]}")

        total_checked = grounded + ungrounded
        if total_checked > 0:
            fidelity = grounded / total_checked * 100
            out.append(f"\n  Fidelidad: {grounded}/{total_checked} frases soportadas ({fidelity:.0f}%)")
            if fidelity >= 80:
                out.append("  Veredicto: RAG fiel")
            elif fidelity >= 50:
                out.append("  Veredicto: Mezcla contexto + conocimiento parametrico")
            else:
                out.append("  Veredicto: LLM con barniz de retrieval (baja fidelidad)")
        else:
            out.append("  (no se pudo evaluar)")

        out.append("=" * 70)
        result = "\n".join(out)
        print(result)
        return result

    def summarize_session_part(
        self,
        session_id: str,
        part: str = "second_half",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        mode: str = "resumen",
        top_k: int = 8,
    ) -> Dict:
        """Summarize a portion of a session using chunks in the time range.

        Args:
            session_id: e.g. "llm_39"
            part: "all", "first_half", "second_half", "first_third",
                  "middle_third", "last_third", "custom"
            start_ms: override start (only with part="custom")
            end_ms: override end (only with part="custom")
            mode: prompt mode (default "resumen")
            top_k: max chunks to include in context
        """
        t0 = time.time()

        range_start, range_end = self.compute_session_range(
            session_id, part=part, start_ms=start_ms, end_ms=end_ms,
        )

        all_chunks = self.get_session_chunks(session_id)
        filtered = self._filter_chunks_by_range(all_chunks, range_start, range_end)

        if not filtered:
            return {
                "response": f"No hay chunks para la sesion '{session_id}' en el rango {format_timestamp(range_start)}-{format_timestamp(range_end)}.",
                "sources": [],
                "timing": {"total_s": round(time.time() - t0, 2)},
            }

        # Sort by quality and take top_k
        filtered.sort(key=lambda c: c.get("quality_score", 0), reverse=True)
        selected = filtered[:top_k]
        # Re-sort by time for coherent context
        selected.sort(key=lambda c: c.get("start_ms", 0))

        context = self.build_context_from_chunks(selected, max_chars=4000)

        query = (
            f"Resume brevemente lo que se explico en {session_id} "
            f"entre {format_timestamp(range_start)} y {format_timestamp(range_end)}"
        )
        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["resumen"])
        user_prompt = self._build_user_prompt(query, context)

        t_llm = time.time()
        response_text = self._call_ollama(system, user_prompt)
        llm_time = time.time() - t_llm

        sources = []
        for c in selected:
            sources.append({
                "topic": c.get("topic", "?"),
                "session_id": c.get("session_id", ""),
                "timestamp": f"{format_timestamp(c.get('start_ms', 0))} - {format_timestamp(c.get('end_ms', 0))}",
                "quality": round(c.get("quality_score", 0), 2),
            })

        return {
            "query": query,
            "mode": mode,
            "session_id": session_id,
            "part": part,
            "range": f"{format_timestamp(range_start)} - {format_timestamp(range_end)}",
            "chunks_available": len(filtered),
            "chunks_used": len(selected),
            "response": response_text,
            "sources": sources,
            "timing": {
                "llm_s": round(llm_time, 2),
                "total_s": round(time.time() - t0, 2),
            },
        }

    # ------------------------------------------------------------------
    # Session notes generation
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_topic_sections(chunks: List[Dict], block_minutes: int = 10) -> List[Dict]:
        """Group chunks into time-based sections, labeled by dominant topics.

        Since pseudo-label topics are very granular (~1 unique topic per chunk),
        we use fixed time blocks and find the most representative topics per block.
        """
        from collections import Counter

        if not chunks:
            return []

        block_ms = block_minutes * 60 * 1000
        sess_start = chunks[0].get("start_ms", 0)
        sections: List[Dict] = []
        current_chunks: List[Dict] = []
        block_end = sess_start + block_ms

        for chunk in chunks:
            if chunk.get("start_ms", 0) >= block_end and current_chunks:
                sections.append(_build_section(current_chunks))
                current_chunks = []
                block_end = chunk.get("start_ms", 0) + block_ms
            current_chunks.append(chunk)

        if current_chunks:
            sections.append(_build_section(current_chunks))

        return sections

    def generate_session_notes(
        self, session_id: str, with_llm: bool = True, max_llm_sections: int = 3,
    ) -> str:
        """Generate structured markdown notes for an entire session.

        Args:
            session_id: e.g. "llm_39"
            with_llm: whether to use Ollama for section summaries
            max_llm_sections: max sections to summarize with LLM (to limit time)

        Returns:
            Markdown string with TOC + sections.
        """
        chunks = self.get_session_chunks(session_id)
        if not chunks:
            return f"No hay datos para la sesion '{session_id}'."

        sections = self._detect_topic_sections(chunks)

        # Build markdown
        total_start = format_timestamp(chunks[0].get("start_ms", 0))
        total_end = format_timestamp(chunks[-1].get("end_ms", 0))
        lines = [
            f"# Apuntes: {session_id}",
            f"**Duracion:** {total_start} - {total_end} | "
            f"**Chunks:** {len(chunks)} | **Secciones:** {len(sections)}",
            "",
            "## Indice",
        ]

        # TOC
        for i, sec in enumerate(sections, 1):
            ts = f"{format_timestamp(sec['start_ms'])} - {format_timestamp(sec['end_ms'])}"
            lines.append(f"{i}. **{sec['topic'].title()}** — `{ts}`")
        lines.append("")

        # Sections
        for i, sec in enumerate(sections, 1):
            ts = f"{format_timestamp(sec['start_ms'])} - {format_timestamp(sec['end_ms'])}"
            lines.append(f"## {i}. {sec['topic'].title()}")
            lines.append(f"*{ts}* — {len(sec['chunks'])} fragmentos\n")

            # Extract key points from chunks
            sec_chunks = sec["chunks"]
            # Sort by quality, pick top ones for bullets
            by_quality = sorted(sec_chunks, key=lambda c: c.get("quality_score", 0), reverse=True)
            top_chunks = by_quality[:5]
            top_chunks.sort(key=lambda c: c.get("start_ms", 0))

            # Collect keywords across the section
            all_keywords: Set[str] = set()
            for c in sec_chunks:
                for kw in c.get("keywords", []):
                    if kw.lower() not in GENERIC_TOPICS:
                        all_keywords.add(kw)
            if all_keywords:
                lines.append(f"**Palabras clave:** {', '.join(sorted(all_keywords)[:10])}\n")

            if with_llm and i <= max_llm_sections:
                # Use Ollama to summarize this section
                context = self.build_context_from_chunks(top_chunks, include_evidence=False, max_chars=3000)
                system = (
                    "Eres un asistente que genera apuntes concisos de clase. "
                    "Genera 3-5 bullet points con los conceptos principales. "
                    "Cita tiempos [MM:SS] cuando sea posible. Se breve y directo."
                )
                user_prompt = (
                    f"Genera apuntes de esta seccion de clase ({sec['topic']}):\n\n{context}"
                )
                try:
                    payload = {
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 500},
                    }
                    resp = requests.post(
                        f"{self.ollama_url}/api/chat", json=payload, timeout=60,
                    )
                    resp.raise_for_status()
                    summary = resp.json()["message"]["content"]
                    lines.append(summary)
                except Exception:
                    # Fallback to extractive bullets
                    for c in top_chunks:
                        ts_c = format_timestamp(c.get("start_ms", 0))
                        preview = c.get("page_content", "")[:200].replace("\n", " ")
                        lines.append(f"- [{ts_c}] {preview}")
            else:
                # No LLM — extractive bullets
                for c in top_chunks:
                    ts_c = format_timestamp(c.get("start_ms", 0))
                    preview = c.get("page_content", "")[:200].replace("\n", " ")
                    lines.append(f"- [{ts_c}] {preview}")

            lines.append("")

        return "\n".join(lines)
