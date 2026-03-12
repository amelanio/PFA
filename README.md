# VideoMind RAG — Proyecto Final

Chatbot RAG local sobre grabaciones de clase con citas temporales (MM:SS).

## Pipeline

1. **Transcripcion** (`notebooks/01_Transcription.ipynb`) — mlx-whisper (large-v3-turbo) sobre MP3
2. **Normalizacion** (`notebooks/02_Transcription_Normalization.ipynb`) — estudio y correccion de errores ASR (alucinaciones, terminos tecnicos, puntuacion)
3. **Pseudo-labels** (`notebooks/03_PseudoLabel_Generation.ipynb`) — Ollama qwen2.5:14b genera JSON estructurado por chunk
4. **Data Preparation** (`notebooks/04_RAG_Data_Preparation.ipynb`) — quality scoring, documentos duales (enriquecido + evidencia)
5. **Indexing** (`notebooks/05_RAG_Indexing.ipynb`) — FAISS (semantico) + BM25 (lexico)
6. **Explorer** (`notebooks/06_RAG_Explorer.ipynb`) — Chatbot Gradio + analisis interactivo

## Estructura

```
notebooks/              Notebooks del pipeline (ejecutar en orden 01-05, 06 es exploracion)
src/rag/                Motor de busqueda (query_engine.py) y UI del chatbot (chat_ui.py)
src/browser_automation/ Grabacion automatizada de clases (scrcpy + Appium)
scripts/                Scripts auxiliares (add_session_to_raw_train, transcribe_sessions)
data/                   Datasets (raw_train, final_train, checkpoints)
outputs/                Transcripciones y estadisticas
docs/                   Documentacion de arquitectura
artifacts/              Generado: audio.json por sesion + rag_docs (gitignored)
.rag_index/             Generado: indices FAISS + BM25 (gitignored)
```

## Requisitos

- Python 3.10+
- Ollama corriendo localmente (`ollama serve`)
- Modelos Ollama: `qwen2.5:14b` (pseudo-labels), `qwen2.5:7b` (chatbot)
- Dependencias: `faiss-cpu`, `sentence-transformers`, `gradio`, `mlx-whisper`

## Uso

```bash
# Lanzar chatbot (tras ejecutar notebooks 01-05)
python src/rag/chat_ui.py

# O desde el notebook 06
jupyter notebook notebooks/06_RAG_Explorer.ipynb
```
