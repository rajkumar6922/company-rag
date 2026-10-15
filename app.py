"""DocLens: upload documents and ask questions about them (Gradio app).

Run with:  uv run python app.py
"""

import html
import logging
import uuid
from pathlib import Path

import gradio as gr

from answer import MissingAPIKeyError, answer_question
from chunker import chunk_text
from loader import SUPPORTED_TYPES, load_text
from store import add_document, create_collection, delete_collection

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("doclens")

# --- Limits (the app is public and uses a paid API key) ----------------------
MAX_FILE_MB = 5
MAX_FILES = 3
MAX_QUESTIONS = 20
MAX_QUESTION_CHARS = 1000
HISTORY_TURNS = 3  # earlier Q&A pairs sent to the model with each question
SESSION_TTL_SECONDS = 3 * 60 * 60  # drop idle sessions after 3 hours

CSS_PATH = Path(__file__).parent / "app.css"

FOOTER = "Files stay in memory for this session only. Retrieved text is sent to OpenAI to write answers."

WELCOME = (
    "### Welcome to DocLens\n"
    "Upload a PDF, Word or text file on the left. "
    "Once it's indexed, ask anything about it and every answer will cite its sources."
)

# --- Theme -------------------------------------------------------------------
# Light values first, *_dark values for dark mode. Matching tokens for custom
# elements live in app.css.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F6F8FB",
    body_background_fill_dark="#070B14",
    body_text_color="#0F172A",
    body_text_color_dark="#E6EDF7",
    body_text_color_subdued="#52627A",
    body_text_color_subdued_dark="#8DA0BD",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#0F172A",
    background_fill_secondary="#F1F5F9",
    background_fill_secondary_dark="#0B1222",
    border_color_primary="rgba(15,23,42,0.12)",
    border_color_primary_dark="rgba(148,163,184,0.18)",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#0F172A",
    block_border_color="rgba(15,23,42,0.12)",
    block_border_color_dark="rgba(148,163,184,0.18)",
    block_label_text_color="#52627A",
    block_label_text_color_dark="#8DA0BD",
    block_title_text_color="#0F172A",
    block_title_text_color_dark="#E6EDF7",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#0B1222",
    input_border_color="rgba(15,23,42,0.14)",
    input_border_color_dark="rgba(148,163,184,0.22)",
    input_border_color_focus="#0891B2",
    input_border_color_focus_dark="#22D3EE",
    button_primary_background_fill="#0891B2",
    button_primary_background_fill_dark="#22D3EE",
    button_primary_background_fill_hover="#0E7490",
    button_primary_background_fill_hover_dark="#67E8F9",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#04121A",
    color_accent="#0891B2",
    color_accent_soft="rgba(8,145,178,0.10)",
    color_accent_soft_dark="rgba(34,211,238,0.10)",
    link_text_color="#0891B2",
    link_text_color_dark="#22D3EE",
)

# Dark by default: Gradio toggles a "dark" class on <body>. Add it unless the
# URL explicitly asks for light mode (?__theme=light).
FORCE_DARK_JS = """
() => {
  const theme = new URLSearchParams(window.location.search).get("__theme");
  if (theme !== "light") document.body.classList.add("dark");
}
"""


# --- Session state -----------------------------------------------------------
def new_session() -> dict:
    """Create the per-browser-session state with its own Chroma collection."""
    session_id = uuid.uuid4().hex
    return {
        "id": session_id,
        "collection": create_collection(session_id),
        "files": {},  # filename -> number of chunks
        "questions": 0,
        "history": [],  # plain {"role","content"} turns sent to the model
        "chat": [],  # what the chatbot shows (answers + Sources line)
        "examples": [],
    }


def ensure_session(session):
    # State starts as None; create the session lazily on first use.
    return session if session else new_session()


def drop_session(session) -> None:
    # Called by Gradio when a session's state expires or the tab closes.
    if session:
        delete_collection(session["id"])


# --- HTML fragments (everything from files or the model is escaped) ----------
LOGO_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">'
    '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>'
    '<path d="M8.5 11h5M11 8.5v5"/></svg>'
)


def hero_html(session) -> str:
    has_files = bool(session and session["files"])
    asked = bool(session and session["questions"])
    steps = [
        ("1", "Upload", "done" if has_files else "active"),
        ("2", "Index", "done" if has_files else ""),
        ("3", "Ask", "done" if asked else ("active" if has_files else "")),
    ]
    sep = '<li class="dl-sep" aria-hidden="true"></li>'
    items = sep.join(
        f'<li class="{state}"><span>{num}</span>{label}</li>' for num, label, state in steps
    )
    return f"""
<header class="dl-hero">
  <div class="dl-brand"><div class="dl-logo">{LOGO_SVG}</div><h1 class="dl-title">DocLens</h1></div>
  <p class="dl-tagline">Upload PDFs, Word files or text. Ask questions. Every answer shows its sources.</p>
  <ol class="dl-steps">{items}</ol>
</header>"""


def docs_html(session) -> str:
    files = session["files"] if session else {}
    if not files:
        return (
            '<div class="dl-section-label">Indexed</div>'
            '<div class="dl-empty">No documents yet.<br>Drop up to '
            f"{MAX_FILES} files above (max {MAX_FILE_MB} MB each).</div>"
        )
    cards = "".join(
        f'<div class="dl-file">'
        f'<span class="dl-file-type">{html.escape(Path(name).suffix.lstrip(".").upper())}</span>'
        f'<span class="dl-file-name" title="{html.escape(name)}">{html.escape(name)}</span>'
        f'<span class="dl-pill">{n} chunk{"s" if n != 1 else ""}</span></div>'
        for name, n in files.items()
    )
    return f'<div class="dl-section-label">Indexed</div><div class="dl-files">{cards}</div>'


def meter(label: str, used: int, limit: int) -> str:
    pct = min(100, round(100 * used / limit))
    full = " full" if used >= limit else ""
    return (
        f'<div class="dl-meter"><div class="dl-meter-head"><span>{label}</span>'
        f"<b>{used} / {limit}</b></div>"
        f'<div class="dl-bar{full}"><i style="width:{pct}%"></i></div></div>'
    )


def meters_html(session) -> str:
    files = len(session["files"]) if session else 0
    questions = session["questions"] if session else 0
    return (
        '<div class="dl-section-label">Usage this session</div><div class="dl-meters">'
        + meter("Files", files, MAX_FILES)
        + meter("Questions", questions, MAX_QUESTIONS)
        + "</div>"
    )


def notices_html(notices: list[tuple[str, str]]) -> str:
    # notices: (kind, message) with kind in ok / warn / error / busy.
    if not notices:
        return ""
    rows = "".join(
        f'<div class="dl-notice {kind}">'
        + ('<span class="dl-spinner"></span>' if kind == "busy" else "")
        + f"<span>{html.escape(msg)}</span></div>"
        for kind, msg in notices
    )
    return f'<div class="dl-notices" role="status">{rows}</div>'


CONTEXT_EMPTY = '<div class="dl-empty">Ask a question to see the passages the answer was based on.</div>'
PREVIEW_CHARS = 240


def context_html(chunks: list[dict]) -> str:
    if not chunks:
        return CONTEXT_EMPTY
    cards = []
    for rank, c in enumerate(chunks, start=1):
        # Cosine distance -> relevance in 0..1 (1 = identical meaning).
        relevance = max(0.0, min(1.0, 1 - c["distance"]))
        text = c["text"]
        if len(text) > PREVIEW_CHARS:
            body = (
                "<details><summary>"
                f'<p class="dl-chunk-text dl-preview">{html.escape(text[:PREVIEW_CHARS].rstrip())}…</p>'
                "</summary>"
                f'<p class="dl-chunk-text dl-full">{html.escape(text)}</p></details>'
            )
        else:
            body = f'<p class="dl-chunk-text">{html.escape(text)}</p>'
        cards.append(
            '<div class="dl-chunk"><div class="dl-chunk-head">'
            f'<span class="dl-chunk-rank">#{rank}</span>'
            f'<span class="dl-chunk-src">{html.escape(c["source"])}</span>'
            f'<span class="dl-chunk-idx">chunk {int(c["chunk_index"])}</span>'
            f'<span class="dl-chunk-score">relevance {relevance:.2f}</span></div>'
            f'<div class="dl-bar"><i style="width:{relevance * 100:.0f}%"></i></div>'
            f"{body}</div>"
        )
    return f'<div class="dl-chunks">{"".join(cards)}</div>'


# --- Example questions ---------------------------------------------------------
def example_questions(text: str) -> list[str]:
    """Three starter questions based on the first file (no LLM call)."""
    # Use the first line as a title if it looks like one.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "").lstrip("# ")
    if 3 <= len(first_line) <= 70 and not first_line.endswith("."):
        first = f"What is “{first_line}” about?"
    else:
        first = "What is this document about?"
    return [
        first,
        "Summarize the key points in 3 bullets.",
        "Which numbers, dates or deadlines are mentioned?",
    ]


def chip_updates(session) -> list:
    examples = session["examples"] if session else []
    return [
        gr.update(value=examples[i], visible=True) if i < len(examples) else gr.update(visible=False)
        for i in range(3)
    ]


# --- Event handlers ------------------------------------------------------------
def index_files(files, session):
    """Validate, load, chunk and store newly dropped files."""
    session = ensure_session(session)
    paths = [Path(f) for f in (files or [])]
    if not paths:
        yield (session, gr.skip(), gr.skip(), "", gr.skip(), gr.skip(), *[gr.skip()] * 3)
        return

    # Loading state while we work.
    busy = notices_html([("busy", f"Indexing {len(paths)} file{'s' if len(paths) != 1 else ''}…")])
    yield (session, gr.skip(), gr.skip(), busy, gr.skip(), gr.skip(), *[gr.skip()] * 3)

    notices = []
    for path in paths:
        name = path.name
        try:
            if path.suffix.lower() not in SUPPORTED_TYPES:
                notices.append(("error", f"{name}: unsupported type. Use {', '.join(SUPPORTED_TYPES)}."))
                continue
            if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                notices.append(("warn", f"{name} is larger than {MAX_FILE_MB} MB. Please upload a smaller file."))
                continue
            is_new = name not in session["files"]
            if is_new and len(session["files"]) >= MAX_FILES:
                notices.append(("warn", f"{name} was skipped: the limit is {MAX_FILES} files per session."))
                continue

            text = load_text(path)
            chunks = chunk_text(text)
            if not chunks:
                notices.append(("warn", f"{name} contains no readable text."))
                continue

            add_document(session["collection"], name, chunks)
            session["files"][name] = len(chunks)
            if not session["examples"]:
                session["examples"] = example_questions(text)
            verb = "Indexed" if is_new else "Re-indexed"
            notices.append(("ok", f"{verb} {name} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})."))
        except ValueError as e:
            # loader.py raises ValueError with a user-friendly message
            # (scanned PDF, unsupported type).
            notices.append(("error", str(e)))
        except Exception:
            log.exception("Failed to index %s", name)
            notices.append(("error", f"Couldn't read {name}. It may be damaged or password-protected."))

    yield (
        session,
        None,  # clear the dropzone; indexed files are shown as cards
        docs_html(session),
        notices_html(notices),
        meters_html(session),
        hero_html(session),
        *chip_updates(session),
    )


def submit_question(question, session):
    """Show the user's message right away and lock the input."""
    session = ensure_session(session)
    question = (question or "").strip()
    if not question:
        return gr.update(), session, session["chat"], "", gr.update()
    session["chat"].append({"role": "user", "content": question})
    return (
        gr.update(value="", interactive=False),
        session,
        session["chat"],
        question,
        gr.update(interactive=False),
    )


def respond(question, session):
    """Answer the pending question (the chatbot shows a typing indicator meanwhile)."""
    session = ensure_session(session)
    chunks = None  # None = leave the context panel unchanged

    if not question:
        reply = None
    elif not session["files"]:
        reply = "Please upload a document first. Drop a PDF, Word or text file on the left."
    elif session["questions"] >= MAX_QUESTIONS:
        reply = f"You've reached the limit of {MAX_QUESTIONS} questions for this session. Refresh the page to start a new one."
    elif len(question) > MAX_QUESTION_CHARS:
        reply = f"That question is too long. Please keep it under {MAX_QUESTION_CHARS} characters."
    else:
        try:
            history = session["history"][-2 * HISTORY_TURNS:]
            answer, sources, chunks = answer_question(session["collection"], question, history=history)
            session["questions"] += 1
            session["history"] += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            # Backticks become amber chips; strip any backticks inside names.
            chips = " ".join(f"`{s.replace('`', '')}`" for s in sources)
            reply = f"{answer}\n\n**Sources:** {chips}" if sources else answer
        except MissingAPIKeyError:
            log.error("OPENAI_API_KEY is not set")
            reply = "DocLens isn't configured yet: the OpenAI API key is missing. Please try again later."
        except Exception:
            log.exception("Answering failed")
            reply = "Sorry, I couldn't get an answer right now. Please try again in a moment."

    if reply:
        session["chat"].append({"role": "assistant", "content": reply})

    return (
        session,
        session["chat"],
        context_html(chunks) if chunks is not None else gr.skip(),
        meters_html(session),
        hero_html(session),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def clear_chat(session):
    """Start a fresh conversation; documents and usage counts stay."""
    session = ensure_session(session)
    session["chat"], session["history"] = [], []
    return session, [], CONTEXT_EMPTY


# --- About tab (static text only) ---------------------------------------------
REPO_URL = "https://github.com/rajkumar6922/company-rag"

ABOUT_STEPS = [
    ("Upload", "PDF, DOCX, TXT, MD"),
    ("Chunk", "about 500 characters, 100 overlap"),
    ("Embed", "all-MiniLM-L6-v2, 384 dimensions"),
    ("Retrieve", "top 5 by cosine similarity in Chroma"),
    ("Answer", "gpt-4.1-nano, only from retrieved context, with sources"),
]

ABOUT_STACK = [
    ("Python 3.12", "language"),
    ("Gradio", "web UI"),
    ("ChromaDB", "vector database (in-memory, one collection per session)"),
    ("all-MiniLM-L6-v2", "embedding model"),
    ("OpenAI gpt-4.1-nano", "answer generation"),
    ("pypdf / python-docx", "document parsing"),
    ("uv", "package management"),
    ("Hugging Face Spaces", "hosting"),
    ("GitHub", "source control"),
]

_ARROW = '<li class="dl-flow-arrow" aria-hidden="true">→</li>'

ABOUT_HTML = f"""
<div class="dl-about">
  <section class="dl-card">
    <h2 class="dl-card-title">Is RAG still alive?</h2>
    <p class="dl-card-text">People keep saying long context windows killed RAG. In practice, retrieval
    just changed shape. Stuffing everything into the prompt is slower and more expensive, and models can
    still miss details buried in the middle of a long input. So RAG lives on in new forms: agentic RAG
    (an AI agent decides when and what to search), retrieval as a tool the model calls, memory systems,
    and context engineering (choosing exactly what goes into the prompt). DocLens is the foundation those
    all build on: chunk, embed, retrieve, answer with sources.</p>
  </section>

  <section class="dl-card">
    <h2 class="dl-card-title">How DocLens works</h2>
    <ol class="dl-flow">{_ARROW.join(
        f'<li class="dl-flow-step"><span class="dl-flow-num">{i}</span>'
        f'<b>{name}</b><small>{detail}</small></li>'
        for i, (name, detail) in enumerate(ABOUT_STEPS, start=1)
    )}</ol>
  </section>

  <section class="dl-card">
    <h2 class="dl-card-title">Tech stack</h2>
    <ul class="dl-stack">{"".join(
        f'<li class="dl-badge"><b>{name}</b><small>{role}</small></li>' for name, role in ABOUT_STACK
    )}</ul>
  </section>

  <p class="dl-about-footer">Built by Rajkumar Bathini ·
    <a href="{REPO_URL}" target="_blank" rel="noopener noreferrer">{REPO_URL.removeprefix("https://")}</a></p>
</div>"""

# --- Layout ----------------------------------------------------------------------
with gr.Blocks(title="DocLens · Ask your documents", analytics_enabled=False) as demo:
    session_state = gr.State(None, time_to_live=SESSION_TTL_SECONDS, delete_callback=drop_session)
    pending = gr.State("")  # question waiting for an answer

    hero = gr.HTML(hero_html(None))

    with gr.Tabs(elem_id="dl-tabs"):
        with gr.Tab("Chat"):
            with gr.Row(elem_id="dl-main", equal_height=False):
                # Left: documents
                with gr.Column(scale=35, min_width=300, elem_classes="dl-panel"):
                    gr.HTML(
                        '<div class="dl-panel-title">Your documents'
                        f"<small>PDF · DOCX · TXT · MD, max {MAX_FILE_MB} MB</small></div>"
                    )
                    dropzone = gr.File(
                        file_count="multiple",
                        file_types=list(SUPPORTED_TYPES),
                        type="filepath",
                        show_label=False,
                        height=150,
                        elem_id="dl-drop",
                    )
                    notices = gr.HTML("", elem_classes="dl-notice-block")
                    docs = gr.HTML(docs_html(None))
                    meters = gr.HTML(meters_html(None))

                # Right: chat
                with gr.Column(scale=65, min_width=320, elem_classes="dl-panel"):
                    gr.HTML('<div class="dl-panel-title">Ask your documents<small>Answers cite their sources</small></div>')
                    with gr.Row(elem_classes="dl-chips"):
                        chips = [gr.Button(visible=False, size="sm", elem_classes="dl-chip") for _ in range(3)]
                    chatbot = gr.Chatbot(
                        value=[],
                        show_label=False,
                        height=460,
                        layout="bubble",
                        placeholder=WELCOME,
                        buttons=["copy"],
                        elem_id="chat",
                    )
                    with gr.Row(elem_classes="dl-inputrow"):
                        question_box = gr.Textbox(
                            placeholder="Ask a question about your documents…",
                            show_label=False,
                            container=False,
                            max_lines=4,
                            scale=8,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)
                        clear_btn = gr.Button("New chat", scale=1, min_width=96, elem_classes="dl-ghost")
                    with gr.Accordion("Retrieved context", open=False, elem_id="dl-context"):
                        context = gr.HTML(CONTEXT_EMPTY)

        with gr.Tab("About"):
            gr.HTML(ABOUT_HTML)

    gr.HTML(f'<footer class="dl-footer">{FOOTER}</footer>')

    # Index as soon as files are dropped.
    dropzone.upload(
        index_files,
        inputs=[dropzone, session_state],
        outputs=[session_state, dropzone, docs, notices, meters, hero, *chips],
        show_progress="hidden",  # we show our own spinner notice
    )

    # Asking: 1) echo the question and lock input, 2) answer. Only the chatbot
    # shows a loading state (its typing indicator) while the answer is written.
    answer_outputs = [session_state, chatbot, context, meters, hero, question_box, ask_btn]
    for trigger in (question_box.submit, ask_btn.click):
        trigger(
            submit_question,
            inputs=[question_box, session_state],
            outputs=[question_box, session_state, chatbot, pending, ask_btn],
            show_progress="hidden",
        ).then(respond, inputs=[pending, session_state], outputs=answer_outputs, show_progress_on=chatbot)

    # Example chips: the button's label is the question.
    for chip in chips:
        chip.click(
            submit_question,
            inputs=[chip, session_state],
            outputs=[question_box, session_state, chatbot, pending, ask_btn],
            show_progress="hidden",
        ).then(respond, inputs=[pending, session_state], outputs=answer_outputs, show_progress_on=chatbot)

    clear_btn.click(clear_chat, inputs=[session_state], outputs=[session_state, chatbot, context])


if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css_paths=[CSS_PATH],
        js=FORCE_DARK_JS,
        max_file_size=f"{MAX_FILE_MB}mb",  # enforced by Gradio before upload too
        show_error=False,  # never show stack traces to users
        footer_links=["gradio", "settings"],
    )
