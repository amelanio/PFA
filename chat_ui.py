"""RAG Chatbot UI — Gradio interface for NotebookRAGEngine.

Launch:
    python src/rag/chat_ui.py

Opens http://localhost:7860
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path when running as script
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
from src.rag.query_engine import NotebookRAGEngine, format_timestamp

# ---------------------------------------------------------------------------
# Video file mapping (session_id → mp4 path)
# ---------------------------------------------------------------------------

RECORDINGS_DIR = _project_root / "recordings"
THUMB_DIR = Path(tempfile.mkdtemp(prefix="rag_thumbs_")).resolve()
AUDIO_DIR = Path(tempfile.mkdtemp(prefix="rag_audio_")).resolve()

# Max audio clips to show in the UI
N_AUDIO_SLOTS = 5


def _build_video_map() -> dict[str, str]:
    """Scan recordings/cropped/ (preferred) or recordings/ and map session IDs like 'llm_38' to video paths."""
    vmap = {}
    cropped_dir = RECORDINGS_DIR / "cropped"
    scan_dir = cropped_dir if cropped_dir.exists() else RECORDINGS_DIR
    if not scan_dir.exists():
        return vmap
    for mp4 in sorted(scan_dir.glob("*.mp4")):
        m = re.match(r"^(\d+)_", mp4.name)
        if m:
            session_id = f"llm_{m.group(1)}"
            vmap[session_id] = str(mp4)
    return vmap


VIDEO_MAP = _build_video_map()


# ---------------------------------------------------------------------------
# Thumbnail + audio clip extraction
# ---------------------------------------------------------------------------

def _ts_to_seconds(ts: str) -> float:
    """Convert 'MM:SS' or 'H:MM:SS' to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def _extract_timestamp_ranges(text: str) -> list[tuple[float, float]]:
    """Find MM:SS - MM:SS ranges in text. Returns [(start_s, end_s), ...]."""
    ranges = re.findall(r"(\d{1,3}:\d{2})\s*[-–]\s*(\d{1,3}:\d{2})", text)
    seen = set()
    result = []
    for start_ts, end_ts in ranges:
        start_s = _ts_to_seconds(start_ts)
        end_s = _ts_to_seconds(end_ts)
        key = (start_s, end_s)
        if key not in seen and end_s > start_s:
            seen.add(key)
            result.append(key)
    return sorted(result)[:N_AUDIO_SLOTS]


def _extract_timestamps(text: str) -> list[float]:
    """Find all MM:SS timestamps in text and return unique seconds, sorted."""
    matches = re.findall(r"(\d{1,3}:\d{2})", text)
    seen = set()
    result = []
    for ts in matches:
        secs = _ts_to_seconds(ts)
        if secs not in seen:
            seen.add(secs)
            result.append(secs)
    return sorted(result)


def _seconds_to_label(secs: float) -> str:
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m}:{s:02d}"


def _generate_thumbnail(video_path: str, seconds: float) -> str | None:
    """Extract a single frame from video at given seconds using ffmpeg."""
    safe_label = _seconds_to_label(seconds).replace(":", "-")
    vid_hash = hash(video_path) % 100000
    out_path = THUMB_DIR / f"thumb_{vid_hash}_{safe_label}.jpg"

    if out_path.exists():
        return str(out_path)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(seconds),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "5",
                "-vf", "scale=320:-1",
                str(out_path),
            ],
            capture_output=True,
            timeout=10,
        )
        if out_path.exists():
            return str(out_path)
    except Exception:
        pass
    return None


def _generate_thumbnails(video_path: str, timestamps: list[float]) -> list[tuple[str, str]]:
    """Generate thumbnails for timestamps. Returns [(path, label), ...]."""
    results = []
    for secs in timestamps[:8]:
        thumb = _generate_thumbnail(video_path, secs)
        if thumb:
            results.append((thumb, _seconds_to_label(secs)))
    return results


def _extract_audio_clip(video_path: str, start_s: float, end_s: float) -> str | None:
    """Extract audio clip from video between start and end seconds."""
    duration = min(end_s - start_s, 120)  # max 2 min
    safe_start = _seconds_to_label(start_s).replace(":", "-")
    safe_end = _seconds_to_label(end_s).replace(":", "-")
    vid_hash = hash(video_path) % 100000
    out_path = AUDIO_DIR / f"clip_{vid_hash}_{safe_start}_to_{safe_end}.mp3"

    if out_path.exists():
        return str(out_path)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_s),
                "-i", video_path,
                "-t", str(duration),
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "6",
                str(out_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if out_path.exists():
            return str(out_path)
    except Exception:
        pass
    return None


def _generate_audio_clips(
    video_path: str, ranges: list[tuple[float, float]]
) -> list[tuple[str, str]]:
    """Generate audio clips for timestamp ranges. Returns [(path, label), ...]."""
    results = []
    for start_s, end_s in ranges[:N_AUDIO_SLOTS]:
        clip = _extract_audio_clip(video_path, start_s, end_s)
        if clip:
            label = f"{_seconds_to_label(start_s)} - {_seconds_to_label(end_s)}"
            results.append((clip, label))
    return results


# ---------------------------------------------------------------------------
# Engine (singleton — loaded once at startup)
# ---------------------------------------------------------------------------

print("Cargando NotebookRAGEngine...")
engine = NotebookRAGEngine()
print(f"Engine listo. Sessions: {engine.get_session_ids()}, FAISS: {engine.faiss_enriched.ntotal} docs")
ollama_ok = engine.check_ollama()
if not ollama_ok:
    print(f"AVISO: Ollama no disponible ({engine.ollama_url}). Solo retrieval funcionara.")

# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "Explicar": "explicar",
    "Tecnico": "tecnico",
    "Examen": "examen",
    "Resumen": "resumen",
    "Apuntes": "apuntes",
}


def _format_sources(sources: list) -> str:
    if not sources:
        return "*Sin fuentes*"
    lines = []
    for i, s in enumerate(sources, 1):
        topic = s.get("topic", "?")
        ts = s.get("timestamp", "?")
        quality = s.get("quality", 0)
        session = s.get("session_id", "")
        score = s.get("score", 0)
        session_tag = f" [{session}]" if session else ""
        lines.append(
            f"**{i}.** {topic}{session_tag} — `{ts}` (calidad: {quality}, score: {score})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based intent classifier (replaces regex detection)
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """Eres un clasificador de intenciones para un chatbot de clases universitarias.
Las sesiones disponibles son: {sessions}

Clasifica la query del usuario en UNA de estas intenciones:
- "pregunta": pregunta sobre un tema (ej: "que es PCA?", "explícame backpropagation")
- "apuntes": pide ver todo el contenido/temas de una sesión (ej: "de qué se habla en llm_39", "apuntes de clase 38", "temas de la sesion 40")
- "temporal": pregunta sobre un momento específico (ej: "minuto 45 de clase 39", "qué se dijo entre 10:00 y 20:00 de llm_38", "del minuto 40 al 60 de llm_38")
- "cross_session": busca un tema en varias sesiones (ej: "en qué clases se habla de embeddings?", "dónde se explica fine-tuning?")
- "session_part": pide resumen de una parte específica (ej: "primera mitad de llm_39", "segunda parte de clase 40")

Responde SOLO con JSON valido, sin explicacion:
{{"intent": "...", "session_id": "llm_XX o null", "start_min": null, "end_min": null, "part": "first_half/second_half/first_third/middle_third/last_third o null"}}

Si mencionan "clase XX" o "sesion XX", convierte a "llm_XX".
Para temporal: extrae start_min y end_min en minutos. Si dice "minuto 45", usa start_min=42, end_min=48 (ventana ±3min).
Si no hay sesion, session_id es null."""

_CLASSIFY_CACHE: dict[str, dict] = {}


def _classify_query(query: str) -> dict:
    """Classify user intent using LLM. Returns dict with intent and params."""
    # Check cache
    cache_key = query.strip().lower()
    if cache_key in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[cache_key]

    sessions = ", ".join(engine.get_session_ids())
    system = _CLASSIFY_SYSTEM.format(sessions=sessions)

    try:
        import requests
        payload = {
            "model": engine.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 150},
        }
        resp = requests.post(
            f"{engine.ollama_url}/api/chat", json=payload, timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        # Extract JSON from response (handle markdown fences)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        # Validate session_id
        sid = result.get("session_id")
        if sid and sid not in engine.get_session_ids():
            result["session_id"] = None

        _CLASSIFY_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"[classify] LLM fallback: {e}")
        return _classify_fallback(query)


def _classify_fallback(query: str) -> dict:
    """Regex fallback when Ollama is unavailable."""
    # Extract session ID
    session_id = None
    m = re.search(r"(?:llm_|clase\s*|sesi[oó]n\s*)(\d+)", query, re.IGNORECASE)
    if m:
        sid = f"llm_{m.group(1)}"
        if sid in engine.get_session_ids():
            session_id = sid

    # Temporal: "minuto N", "MM:SS", "entre X y Y"
    m_range = re.search(
        r"entre\s+(\d{1,3}):(\d{2})\s+y\s+(\d{1,3}):(\d{2})", query, re.IGNORECASE,
    )
    if m_range and session_id:
        return {
            "intent": "temporal", "session_id": session_id,
            "start_min": int(m_range.group(1)) + int(m_range.group(2)) / 60,
            "end_min": int(m_range.group(3)) + int(m_range.group(4)) / 60,
            "part": None,
        }

    m_range2 = re.search(
        r"(?:del\s+)?minuto\s+(\d+)\s+al\s+(\d+)", query, re.IGNORECASE,
    )
    if m_range2 and session_id:
        return {
            "intent": "temporal", "session_id": session_id,
            "start_min": int(m_range2.group(1)),
            "end_min": int(m_range2.group(2)),
            "part": None,
        }

    m_min = re.search(r"minuto\s+(\d+)", query, re.IGNORECASE)
    if m_min and session_id:
        center = int(m_min.group(1))
        return {
            "intent": "temporal", "session_id": session_id,
            "start_min": max(0, center - 3), "end_min": center + 3,
            "part": None,
        }

    m_ts = re.search(r"(\d{1,3}):(\d{2})\s+de", query, re.IGNORECASE)
    if m_ts and session_id:
        center = int(m_ts.group(1)) + int(m_ts.group(2)) / 60
        return {
            "intent": "temporal", "session_id": session_id,
            "start_min": max(0, center - 3), "end_min": center + 3,
            "part": None,
        }

    # Session part
    m_part = re.search(r"(primera|segunda|ultima|última)\s+(parte|mitad|tercio)", query, re.IGNORECASE)
    if m_part and session_id:
        pw = m_part.group(1).lower()
        div = m_part.group(2).lower()
        if div in ("parte", "mitad"):
            part = "first_half" if pw == "primera" else "second_half"
        elif div == "tercio":
            part = {"primera": "first_third", "segunda": "middle_third"}.get(pw, "last_third")
        else:
            part = "second_half"
        return {"intent": "session_part", "session_id": session_id, "start_min": None, "end_min": None, "part": part}

    # Cross-session
    if re.search(r"en qu[eé] (sesiones|clases)|d[oó]nde se (habla|explica)|en todas las", query, re.IGNORECASE):
        return {"intent": "cross_session", "session_id": None, "start_min": None, "end_min": None, "part": None}

    # Session-level (apuntes or question)
    if session_id:
        return {"intent": "apuntes", "session_id": session_id, "start_min": None, "end_min": None, "part": None}

    return {"intent": "pregunta", "session_id": None, "start_min": None, "end_min": None, "part": None}


def _format_cross_session_results(result: dict) -> str:
    """Format cross_session_search results as markdown."""
    sessions = result.get("sessions", [])
    total = result.get("total_hits", 0)
    if not sessions:
        return "No encontre resultados relevantes en ninguna sesion."

    lines = [f"**{total} resultados en {len(sessions)} sesiones:**\n"]
    for s in sessions:
        topics_str = ", ".join(s["topics"][:5]) if s["topics"] else "—"
        lines.append(
            f"- **{s['session_id']}** — {s['count']} hits "
            f"(score promedio: {s['avg_score']}) — rango: `{s['time_range']}`\n"
            f"  Temas: {topics_str}"
        )
    return "\n".join(lines)


def _detect_primary_session(sources: list) -> str | None:
    from collections import Counter

    sessions = [s.get("session_id", "") for s in sources if s.get("session_id")]
    if not sessions:
        return None
    most_common = Counter(sessions).most_common(1)[0][0]
    return most_common if most_common in VIDEO_MAP else None


def respond(message: str, history: list, mode_label: str):
    """Process user message and return (response_text, sources_md, primary_session)."""
    mode = MODE_LABELS.get(mode_label, "explicar")

    if not message.strip():
        return "", "*Escribe una pregunta.*", None

    # Classify intent (LLM if available, regex fallback otherwise)
    if ollama_ok:
        intent = _classify_query(message)
    else:
        intent = _classify_fallback(message)

    # Override: if user selected "Apuntes" mode and there's a session, force apuntes
    if mode == "apuntes" and intent.get("session_id"):
        intent["intent"] = "apuntes"

    intent_type = intent.get("intent", "pregunta")
    session_id = intent.get("session_id")
    vid_session = session_id if session_id and session_id in VIDEO_MAP else None

    # --- Route by intent ---

    if intent_type == "cross_session":
        result = engine.cross_session_search(message, top_k=20)
        return _format_cross_session_results(result), "", None

    if intent_type == "temporal" and session_id:
        start_min = intent.get("start_min") or 0
        end_min = intent.get("end_min") or 0
        start_ms = int(start_min * 60 * 1000)
        end_ms = int(end_min * 60 * 1000)
        if not ollama_ok:
            return (
                f"Ollama no disponible. No puedo resumir {session_id} en ese rango.",
                "*Inicia Ollama para usar esta funcion.*", vid_session,
            )
        result = engine.summarize_session_part(
            session_id, part="custom", start_ms=start_ms, end_ms=end_ms, mode=mode,
        )
        response_text = result.get("response", "Sin respuesta.")
        sources = result.get("sources", [])
        timing = result.get("timing", {})
        time_range = result.get("range", "")
        header = f"**{session_id}** — rango: `{time_range}`\n\n"
        timing_note = ""
        if timing:
            timing_note = f"\n\n---\n*Tiempo: total {timing.get('total_s', '?')}s*"
        return header + response_text + timing_note, _format_sources(sources), vid_session

    if intent_type == "apuntes" and session_id:
        result = engine.generate_session_notes(session_id, with_llm=ollama_ok)
        return result, "", vid_session

    if intent_type == "session_part" and session_id:
        part = intent.get("part", "second_half")
        if not ollama_ok:
            return (
                f"Ollama no disponible. No puedo resumir {session_id} ({part}).",
                "*Inicia Ollama para usar esta funcion.*", vid_session,
            )
        result = engine.summarize_session_part(session_id, part=part, mode=mode)

    elif ollama_ok:
        result = engine.ask(message, mode=mode, top_k=5)
    else:
        sources = engine.search_only(message, top_k=5)
        response = "**Ollama no disponible** — mostrando solo resultados de busqueda:\n\n"
        for s in sources:
            response += f"- **{s['topic']}** (`{s['timestamp']}`) — {s['content_preview'][:150]}...\n"
        return response, _format_sources(sources), _detect_primary_session(sources)

    response_text = result.get("response", "Sin respuesta.")
    sources = result.get("sources", [])
    timing = result.get("timing", {})

    timing_note = ""
    if timing:
        parts = []
        if "retrieval_s" in timing:
            parts.append(f"retrieval: {timing['retrieval_s']}s")
        if "llm_s" in timing:
            parts.append(f"LLM: {timing['llm_s']}s")
        if "total_s" in timing:
            parts.append(f"total: {timing['total_s']}s")
        timing_note = f"\n\n---\n*Tiempo: {', '.join(parts)}*"

    primary_session = _detect_primary_session(sources)
    return response_text + timing_note, _format_sources(sources), primary_session


def chat_fn(message: str, history: list, mode_label: str, current_video: str | None):
    """Gradio chat handler — returns all UI outputs."""
    response, sources_md, primary_session = respond(message, history, mode_label)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    # Determine video path
    video_path = current_video
    if primary_session and VIDEO_MAP.get(primary_session):
        candidate = VIDEO_MAP[primary_session]
        if candidate != current_video:
            video_path = candidate

    # Extract timestamps and ranges
    all_text = response + "\n" + sources_md
    timestamps = _extract_timestamps(all_text)
    ranges = _extract_timestamp_ranges(all_text)

    # Generate thumbnails
    thumbs = []
    ts_list = []
    if video_path and timestamps:
        thumbs = _generate_thumbnails(video_path, timestamps)
        ts_list = list(timestamps[:8])

    # Generate audio clips
    audio_clips = []
    if video_path and ranges:
        audio_clips = _generate_audio_clips(video_path, ranges)

    # Build outputs for audio slots: (path_or_None, label)
    audio_outputs = []
    for i in range(N_AUDIO_SLOTS):
        if i < len(audio_clips):
            clip_path, label = audio_clips[i]
            audio_outputs.append(clip_path)
            audio_outputs.append(f"**{label}**")
        else:
            audio_outputs.append(None)
            audio_outputs.append("")

    return [history, sources_md, "", video_path, thumbs, ts_list] + audio_outputs


def on_thumb_select(evt: gr.SelectData, ts_state: list):
    """When a thumbnail is clicked, return the corresponding seconds to seek."""
    if ts_state and 0 <= evt.index < len(ts_state):
        return ts_state[evt.index]
    return None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _load_video(session_id: str) -> str | None:
    return VIDEO_MAP.get(session_id)


def build_ui() -> gr.Blocks:
    video_sessions = sorted(VIDEO_MAP.keys())

    with gr.Blocks(title="RAG Chatbot — VideoMind") as demo:
        ts_state = gr.State([])

        gr.Markdown(
            "# RAG Chatbot — VideoMind\n"
            "Pregunta sobre las clases indexadas. "
            "Miniaturas + clips de audio se generan para cada respuesta."
        )

        with gr.Row():
            # --- Left: Chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=400)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Escribe tu pregunta aqui...",
                        label="Pregunta",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)

                mode = gr.Radio(
                    choices=list(MODE_LABELS.keys()),
                    value="Explicar",
                    label="Modo de respuesta",
                )

            # --- Right: Video + Thumbnails + Audio + Sources ---
            with gr.Column(scale=2):
                video_selector = gr.Dropdown(
                    choices=video_sessions,
                    value=video_sessions[0] if video_sessions else None,
                    label="Clase / Video",
                )
                video_player = gr.Video(
                    label="Video",
                    elem_id="video_player",
                    value=VIDEO_MAP.get(video_sessions[0]) if video_sessions else None,
                    height=300,
                )

                gr.Markdown("### Momentos clave")
                gallery = gr.Gallery(
                    label="Haz clic en una miniatura para saltar",
                    columns=4,
                    height=160,
                    object_fit="cover",
                )
                seek_time = gr.Number(visible=False, elem_id="seek_time_input")

                gr.Markdown("### Audio de las fuentes")
                audio_players = []
                audio_labels = []
                for i in range(N_AUDIO_SLOTS):
                    lbl = gr.Markdown(value="")
                    player = gr.Audio(
                        type="filepath",
                        interactive=False,
                        show_label=False,
                    )
                    audio_players.append(player)
                    audio_labels.append(lbl)

                gr.Markdown("### Fuentes")
                sources_md = gr.Markdown(
                    value="*Las fuentes del ultimo resultado apareceran aqui.*",
                )
                gr.Markdown(
                    f"**Sessions:** {', '.join(engine.get_session_ids())}  \n"
                    f"**Docs:** {engine.faiss_enriched.ntotal} | "
                    f"**Ollama:** {'OK' if ollama_ok else 'No'} | "
                    f"**Modelo:** {engine.ollama_model}"
                )

        # --- Build outputs list for chat_fn ---
        chat_outputs = [chatbot, sources_md, msg, video_player, gallery, ts_state]
        for i in range(N_AUDIO_SLOTS):
            chat_outputs.extend([audio_players[i], audio_labels[i]])

        # --- Events ---

        video_selector.change(
            fn=_load_video,
            inputs=[video_selector],
            outputs=[video_player],
        )

        send_btn.click(
            fn=chat_fn,
            inputs=[msg, chatbot, mode, video_player],
            outputs=chat_outputs,
        )
        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbot, mode, video_player],
            outputs=chat_outputs,
        )

        # Thumbnail click → seek video
        gallery.select(
            fn=on_thumb_select,
            inputs=[ts_state],
            outputs=[seek_time],
        ).then(
            fn=None,
            inputs=[seek_time],
            js="(s) => { const v = document.querySelector('#video_player video'); if (v && s != null) { v.currentTime = s; v.play(); } }",
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    # .resolve() needed on macOS where /tmp → /private/var/folders/...
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[
            str(RECORDINGS_DIR.resolve()),
            str(THUMB_DIR),
            str(AUDIO_DIR),
        ],
    )
