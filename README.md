# VideoMind RAG — Proyecto Final

Chatbot RAG local sobre grabaciones de clase con citas temporales (MM:SS).

## Pipeline

1. **Transcripcion** (`notebooks/01_Transcription.ipynb`) — mlx-whisper sobre MP3
2. **Pseudo-labels** (`notebooks/02_PseudoLabel_Generation.ipynb`) — Ollama qwen2.5:14b genera JSON estructurado
3. **Data Preparation** (`notebooks/03_RAG_Data_Preparation.ipynb`) — quality scoring, documentos duales (enriquecido + evidencia)
4. **Indexing** (`notebooks/04_RAG_Indexing.ipynb`) — FAISS (semantico) + BM25 (lexico)
5. **Chatbot** (`src/rag/chat_ui.py`) — Gradio, retrieval hibrido, generacion con Ollama

## Estructura

```
notebooks/          Notebooks del pipeline (ejecutar en orden)
scripts/            Scripts de procesamiento y automatizacion
src/rag/            Motor de busqueda y UI del chatbot
src/browser_automation/  Grabacion automatizada de clases
data/               Datasets (raw, final, checkpoints)
outputs/            Transcripciones y estadisticas
docs/               Documentacion de arquitectura
```

## Requisitos

- Python 3.10+
- Ollama corriendo localmente (`ollama serve`)
- Dependencias: `faiss-cpu`, `sentence-transformers`, `gradio`, `whisper`

## Uso

```bash
# Lanzar chatbot (tras ejecutar notebooks 01-04)
python3 src/rag/chat_ui.py
```
