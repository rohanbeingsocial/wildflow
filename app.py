# ============================================================================
#  WILDFLOW — STUDY APP (auto-routing retrieval engine + React web UI)
# ============================================================================
#  The student-facing product. Students pick only COLLEGE + YEAR, then just
#  type. Every message is routed automatically:
#
#    1. Gemini (cheap model) classifies the subject + extracts the core topic
#       + decides THEORY vs NUMERICAL  (one call, JSON out)
#    2. The topic is matched against each unit's heading list (deterministic
#       JSON-routing layer from the paper) -> best unit(s) chosen
#    3. Vector search runs ONLY inside that unit's Qdrant collection
#    4. Matched sections are re-read VERBATIM from the unit .md (equations +
#       image refs intact) and given to Gemini as the only source of truth
#    5. Numericals go to the stronger model with a formula-first prompt
#       (falls back to flash if quota runs out); answers are cached briefly
#
#  Run:   python app.py      ->  http://127.0.0.1:8000
# ============================================================================

import os
os.environ.setdefault("USE_TF", "0")   # skip transformers' TensorFlow backend (Keras 3 conflict)

import re
import json
import time
import base64
import hashlib
import difflib
import threading
from pathlib import Path

import numpy as np

import torch
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# ----------------------------------------------------------------------------
# 🔹 CONFIGURATION
# ----------------------------------------------------------------------------
COLLEGES = {
    "MIT-WPU": Path(os.environ.get(
        "WILDFLOW_CONTENT_ROOT",
        r"C:\Users\rohan\Desktop\LLMstudy\MITwpuEmbeddedMD")),
}

API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_API_KEY_HERE")

# Rolling aliases so the app never breaks when Google retires a model version.
# Each list is a quota-fallback chain: if a model 429s, the next one answers.
ROUTER_CHAIN   = ["gemini-flash-latest", "gemini-flash-lite-latest"]
THEORY_CHAIN   = ["gemini-flash-latest", "gemini-flash-lite-latest"]
MATH_CHAIN     = ["gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"]
IMAGE_MODEL    = "gemini-3.1-flash-image"   # stage 2 of graphical queries: steps -> diagram
TEXT_MODEL_NAME = "intfloat/multilingual-e5-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM  = 1536
TOP_K          = 4
MAX_UNITS_SEARCHED = 6          # candidate units; the vector search picks the winner
CONTEXT_CHAR_LIMIT = 14000
CACHE_TTL_SECONDS  = 3600
HOST = os.environ.get("WILDFLOW_HOST", "127.0.0.1")
PORT = int(os.environ.get("WILDFLOW_PORT", "8000"))

STATIC_DIR = Path(__file__).parent / "static"
FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
GENERATED_DIR = Path(__file__).parent / "generated"
SEMANTIC_CACHE_SIM = 0.95     # cosine similarity to reuse a cached answer (paper §7.3)
YEAR_RE = re.compile(r"^(First|Second|Third|Fourth|Final)\s+Year$", re.I)

# ----------------------------------------------------------------------------
# 🔹 Markdown section parser (identical to embed.py)
# ----------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{2,6})\s*(.+?)\s*$", re.MULTILINE)

def clean_for_embedding(text):
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def parse_sections(md_content):
    sections = []
    matches = list(HEADING_RE.finditer(md_content))
    pre_end = matches[0].start() if matches else len(md_content)
    pre_clean = clean_for_embedding(md_content[:pre_end])
    if pre_clean:
        sections.append({"topic": "Introduction", "level": 2,
                         "raw": md_content[:pre_end].strip(), "content": pre_clean})
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_content)
        content = clean_for_embedding(md_content[m.end():end])
        if not content:
            continue
        sections.append({
            "topic": m.group(2).lstrip("#").strip(),
            "level": len(m.group(1)),
            "raw": md_content[m.start():end].strip(),
            "content": content,
        })
    return sections

def norm_topic(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()

# ----------------------------------------------------------------------------
# 🔹 Unit discovery — a unit is any folder with exactly one .md and a Qdrant
#    store (a subfolder containing meta.json with a 1536D collection).
#    Sections are parsed once at scan time (restart the app after re-embedding).
# ----------------------------------------------------------------------------
UNITS = {}  # uid -> {folder, md_path, store, collection, college, year, subject,
            #         name, sections, topic_map}

def find_store(folder: Path):
    for child in folder.iterdir():
        meta = child / "meta.json"
        if child.is_dir() and meta.exists():
            try:
                cfg = json.loads(meta.read_text(encoding="utf-8"))
                for cname, ccfg in cfg.get("collections", {}).items():
                    if ccfg.get("vectors", {}).get("size") == EMBEDDING_DIM:
                        return child, cname
            except Exception:
                continue
    return None

def display_name(folder_name):
    if folder_name.isdigit():
        return f"Lecture {int(folder_name)}"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", folder_name)   # drop trailing "(Subject)"
    return name.replace("_", " — ").strip()

def scan_units():
    UNITS.clear()
    for college, root in COLLEGES.items():
        if not root.is_dir():
            print(f"⚠️  Content root missing for {college}: {root}")
            continue
        for md_path in sorted(root.rglob("*.md")):
            folder = md_path.parent
            if len(list(folder.glob("*.md"))) != 1:
                continue
            store = find_store(folder)
            if not store:
                continue
            rel = folder.relative_to(root)
            parts = rel.parts
            subject = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else "General")
            year = next((p for p in parts if YEAR_RE.match(p)), "")
            uid = hashlib.sha1(f"{college}|{rel}".encode("utf-8")).hexdigest()[:12]
            if uid in UNITS:
                continue
            try:
                sections = parse_sections(md_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            UNITS[uid] = {
                "folder": folder, "md_path": md_path,
                "store": store[0], "collection": store[1],
                "college": college, "year": year, "subject": subject,
                "name": display_name(folder.name),
                "sections": sections,
                "topic_map": {norm_topic(s["topic"]): s for s in sections},
            }
    print(f"✅ Found {len(UNITS)} units across {len(COLLEGES)} college(s)")

def units_in_scope(college, year):
    return {uid: u for uid, u in UNITS.items()
            if u["college"] == college and (not u["year"] or not year or u["year"] == year)}

# ----------------------------------------------------------------------------
# 🔹 Embedding models (loaded lazily, once) + per-unit Qdrant clients
# ----------------------------------------------------------------------------
_models = {}
_models_lock = threading.Lock()

def _free_commit_gb():
    """Available commit charge (physical + pagefile) in GB, Windows only."""
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in
                        ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                         "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPageFile / (1024 ** 3)
    except Exception:
        return 99.0  # non-Windows or check failed: don't block the load

MODEL_LOAD_NEEDS_GB = 5.5   # loading past this margin hard-crashes the process

def get_models():
    with _models_lock:
        if not _models:
            free = _free_commit_gb()
            if free < MODEL_LOAD_NEEDS_GB:
                raise RuntimeError(
                    f"Not enough free memory to load the AI models ({free:.1f} GB free, "
                    f"~{MODEL_LOAD_NEEDS_GB:.0f} GB needed). Close some applications "
                    f"(browser windows, Docker, other tools) and ask again.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⏳ Loading embedding models on {device} (first query only) ...")
            _models["device"] = device
            _models["text"] = SentenceTransformer(TEXT_MODEL_NAME, device=device,
                                                  model_kwargs={"low_cpu_mem_usage": True})
            _models["clip"] = CLIPModel.from_pretrained(CLIP_MODEL_NAME, low_cpu_mem_usage=True).to(device)
            _models["clip_proc"] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            print("✅ Embedding models ready.")
    return _models

def embed_query(text):
    m = get_models()
    sent = m["text"].encode(text, show_progress_bar=False).tolist()
    inputs = m["clip_proc"](text=[text], return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(m["device"])
    with torch.no_grad():
        clip_vec = m["clip"].get_text_features(**inputs).cpu().numpy()[0].tolist()
    return sent + clip_vec

_clients = {}
_clients_lock = threading.Lock()

def get_client(uid):
    unit = UNITS[uid]
    with _clients_lock:
        if uid not in _clients:
            _clients[uid] = QdrantClient(path=str(unit["store"]))
    return _clients[uid]

def section_for_topic(unit, topic):
    key = norm_topic(topic)
    tm = unit["topic_map"]
    if key in tm:
        return tm[key]
    for k, s in tm.items():
        if key and (key in k or k in key):
            return s
    close = difflib.get_close_matches(key, list(tm.keys()), n=1, cutoff=0.7)
    return tm[close[0]] if close else None

# ----------------------------------------------------------------------------
# 🔹 ROUTING — subject via Gemini, unit via topic-list matching (paper §11.2)
# ----------------------------------------------------------------------------
genai.configure(api_key=API_KEY)

def generate_with_fallback(prompt, chain, generation_config=None):
    """Try each model in the chain; quota/availability errors move to the next."""
    last_err = None
    for name in chain:
        try:
            return genai.GenerativeModel(name).generate_content(
                prompt, generation_config=generation_config).text
        except Exception as e:
            last_err = e
    raise last_err

CALC_WORDS = ("calculate", "find", "determine", "compute", "solve", "evaluate",
              "how much", "how many", "what is the value")
DRAW_WORDS = ("draw", "projection", "isometric", "orthographic", "sketch",
              "construct the", "front view", "top view", "side view")

def local_query_type(message):
    """Offline fallback when the router model is unavailable."""
    msg = message.lower()
    if any(w in msg for w in DRAW_WORDS):
        return "GRAPHICAL"
    has_number = bool(re.search(r"\d", msg))
    asks_calc = any(w in msg for w in CALC_WORDS)
    return "NUMERICAL" if (has_number and asks_calc) else "THEORY"

def gemini_route(message, subjects, last_unit_name):
    """One cheap call: subject + core topic + query type + follow-up flag."""
    prompt = f"""You are the query router of a college study assistant.

Available subjects: {json.dumps(subjects)}
Previous unit discussed: {json.dumps(last_unit_name or None)}

Student message: {json.dumps(message)}

Reply with ONLY a JSON object, no other text:
{{"subject": "<exactly one subject from the list, or FOLLOWUP if the message continues the previous discussion without a new topic, or UNKNOWN if unrelated to every subject>",
 "topic": "<the core academic topic in 2-6 words>",
 "query_type": "<NUMERICAL if it requires computing a numeric/symbolic result from given values, GRAPHICAL if it asks to draw/construct/project an engineering drawing, else THEORY>"}}

Examples:
- "A force of 100 N acts at 30 degrees, find its components" -> query_type NUMERICAL, topic "resolution of forces"
- "what is number average molecular mass" -> query_type THEORY, topic "number average molecular mass"
- "Draw the projections of a point 30 mm above HP" -> query_type GRAPHICAL, topic "projection of points"
- "explain that again more simply" -> subject FOLLOWUP"""
    try:
        text = generate_with_fallback(
            prompt, ROUTER_CHAIN,
            generation_config=genai.types.GenerationConfig(temperature=0))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    qt = str(data.get("query_type", "")).upper()
    return {
        "subject": data.get("subject", "UNKNOWN"),
        "topic": data.get("topic", message[:60]),
        "query_type": qt if qt in ("NUMERICAL", "THEORY", "GRAPHICAL") else local_query_type(message),
    }

STOP_TOKENS = {"what", "is", "are", "the", "a", "an", "of", "and", "or", "in", "on",
               "for", "to", "how", "why", "does", "do", "with", "its", "it", "this",
               "explain", "define", "describe", "state", "give", "me", "about", "between"}

def content_tokens(text):
    return {t for t in norm_topic(text).split() if t not in STOP_TOKENS and not t.isdigit()}

def unit_topic_score(text, unit):
    """How well a query/topic matches this unit's heading list (0..1)."""
    q = norm_topic(text)
    q_tokens = content_tokens(text)
    best = 0.0
    for key in unit["topic_map"]:
        t_tokens = content_tokens(key)
        if not t_tokens:
            continue
        overlap = len(q_tokens & t_tokens) / max(1, min(len(q_tokens), len(t_tokens)))
        ratio = difflib.SequenceMatcher(None, q, key).ratio()
        best = max(best, 0.7 * overlap + 0.3 * ratio)
    return best

def rank_units(scope, subject, topic, message):
    cands = {uid: u for uid, u in scope.items() if u["subject"] == subject} or scope
    scored = sorted(
        ((max(unit_topic_score(topic, u), unit_topic_score(message, u)), uid)
         for uid, u in cands.items()),
        reverse=True)
    return [uid for _, uid in scored[:MAX_UNITS_SEARCHED]], (scored[0][0] if scored else 0.0)

# ----------------------------------------------------------------------------
# 🔹 Retrieval + generation
# ----------------------------------------------------------------------------
def section_images(uid, text, limit=6):
    """Existing diagram files referenced inside a Markdown slice -> served URLs."""
    unit = UNITS[uid]
    urls = []
    for fname in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
        fname = fname.split("/")[-1].split("\\")[-1]
        if (unit["folder"] / fname).is_file():
            url = f"/api/media?u={uid}&f={fname}"
            if url not in urls:
                urls.append(url)
        if len(urls) >= limit:
            break
    return urls

def retrieve(unit_ids, question, qvec=None):
    """Vector-search the routed unit(s); re-read hits from the live Markdown."""
    if qvec is None:
        qvec = embed_query(question)
    all_hits = []
    for uid in unit_ids:
        unit = UNITS[uid]
        try:
            hits = get_client(uid).search(collection_name=unit["collection"],
                                          query_vector=qvec, limit=TOP_K, with_payload=True)
            all_hits.extend((hit, uid) for hit in hits)
        except Exception as e:
            print(f"⚠️  Search failed in {unit['name']}: {e}")
    all_hits.sort(key=lambda h: h[0].score, reverse=True)

    chosen, topics, sources, seen = [], [], [], set()
    best_uid = unit_ids[0] if unit_ids else None
    for i, (hit, uid) in enumerate(all_hits[:TOP_K]):
        if i == 0:
            best_uid = uid
        payload = hit.payload or {}
        topic = payload.get("topic") or "Untitled"
        section = section_for_topic(UNITS[uid], topic)
        text = section["raw"] if section else (
            payload.get("content") or payload.get("text") or payload.get("text_chunk") or "")
        label = section["topic"] if section else topic
        if not text or (uid, label) in seen:
            continue
        seen.add((uid, label))
        chosen.append((text, uid))
        topics.append({"topic": label, "score": round(float(hit.score), 3)})
        sources.append({"topic": label, "unit": UNITS[uid]["name"], "text": text[:3500]})

    context, gallery = "", []
    for text, uid in chosen:
        if len(context) + len(text) > CONTEXT_CHAR_LIMIT:
            break
        context += text + "\n\n---\n\n"
        for url in section_images(uid, text):
            if url not in gallery and len(gallery) < 6:
                gallery.append(url)
    return context.strip(), topics, best_uid, sources, gallery

def build_prompt(question, context, unit, query_type, history):
    convo = ""
    if history:
        turns = []
        for h in history[-4:]:
            role = "Student" if h.get("role") == "user" else "Tutor"
            turns.append(f"{role}: {h.get('text', '')[:600]}")
        convo = "Recent conversation (for follow-up context only):\n" + "\n".join(turns) + "\n\n"

    prompt = f"""You are Wildflow, a study tutor for the unit "{unit['name']}" ({unit['subject']}).
The COURSE MATERIAL below is extracted verbatim from the official unit notes and is your ONLY source of truth.

STRICT RULES:
- Answer using ONLY the course material. Do not silently mix in outside knowledge.
- If the material does not cover the question, say so plainly first. You may then add
  a short section titled "**Beyond your notes:**" clearly marked as outside the syllabus.
- Keep every equation in display math exactly as written, wrapped in $$ ... $$.
- When a diagram in the material is relevant, include its reference EXACTLY as it
  appears, e.g. ![](_page_5_Figure_15.jpeg) on its own line. Never invent image names.
- Format with Markdown: short headings, bullet points, bold key terms. Write like a
  good teacher preparing the student for an exam — clear, structured, no fluff.

{convo}COURSE MATERIAL:
=====
{context if context else "(nothing relevant was found in this unit)"}
=====

Student's question: {question}
"""
    if query_type == "NUMERICAL":
        prompt += """
This is a NUMERICAL problem. Solve it formula-first:
1. **Formula** — state the exact formula from the course material (in $$ ... $$).
2. **Given** — list the given values with units.
3. **Substitution** — substitute values into the formula.
4. **Steps** — show each calculation step with units.
5. **Answer** — final result, bolded, with units.
If the required formula is not in the course material, say so before solving.
"""
    elif query_type == "GRAPHICAL":
        prompt += """
This is a GRAPHICAL / engineering-drawing problem. Produce:
1. **Given** — the data from the problem statement.
2. **Construction steps** — precise, numbered geometric steps a student can follow
   on paper (reference lines, distances in mm, projection lines, labels).
3. **Result** — one sentence describing what the finished drawing shows.
Keep the steps unambiguous; a student should be able to reproduce the drawing exactly.
"""
    return prompt

def verify_numerical(answer):
    """Re-compute the solution's final arithmetic locally with sympy (paper §8.2)."""
    try:
        text = generate_with_fallback(
            "From this worked solution, extract ONLY the final numeric computation.\n"
            'Reply with ONLY JSON: {"expr": "<the arithmetic expression, using numbers and '
            '+ - * / ** ( ) sqrt() pi only>", "result": <the final numeric value the solution claims>}\n'
            'If there is no clear final computation, reply {"expr": "", "result": null}\n\n'
            "SOLUTION:\n" + answer[-3000:],
            ROUTER_CHAIN, generation_config=genai.types.GenerationConfig(temperature=0))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        expr = str(data.get("expr") or "").replace("^", "**").replace(",", "")
        claimed = data.get("result")
        if not expr or claimed is None:
            return {"status": "unknown"}
        if not re.fullmatch(r"[\d\s+\-*/().eE]*", expr.replace("sqrt", "").replace("pi", "")):
            return {"status": "unknown"}
        import sympy
        computed = float(sympy.sympify(expr))
        claimed = float(claimed)
        ok = abs(computed - claimed) <= max(1e-6, 0.02 * abs(computed))
        return {"status": "verified" if ok else "mismatch",
                "computed": round(computed, 6), "claimed": claimed}
    except Exception:
        return {"status": "unknown"}

def generate_diagram(description):
    """Stage 2 of graphical queries: construction steps -> generated diagram."""
    try:
        GENERATED_DIR.mkdir(exist_ok=True)
        resp = genai.GenerativeModel(IMAGE_MODEL).generate_content(
            "Generate a clean engineering-drawing style diagram: thin black lines on a white "
            "background, labeled points, dimension markings, no shading, no decorative elements. "
            "Draw exactly this construction:\n" + description[:2000])
        for part in resp.candidates[0].content.parts:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None) if blob else None
            if data:
                if isinstance(data, str):
                    data = base64.b64decode(data)
                ext = "png" if "png" in str(getattr(blob, "mime_type", "png")) else "jpg"
                fname = f"{hashlib.sha1(data[:256]).hexdigest()[:16]}.{ext}"
                (GENERATED_DIR / fname).write_bytes(data)
                return f"/api/gen/{fname}"
    except Exception as e:
        print(f"⚠️  Diagram generation failed: {type(e).__name__}: {e}")
    return None

def generate_answer(question, context, unit, query_type, history):
    prompt = build_prompt(question, context, unit, query_type, history)
    chain = MATH_CHAIN if query_type == "NUMERICAL" else THEORY_CHAIN
    return generate_with_fallback(prompt, chain)

# ----------------------------------------------------------------------------
# 🔹 Answer post-processing: local image refs -> served URLs
# ----------------------------------------------------------------------------
IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

def rewrite_images(answer, uid):
    unit = UNITS[uid]
    used = []
    def repl(m):
        if m.group(2).startswith(("http://", "https://", "/api/")):
            return m.group(0)
        fname = m.group(2).split("/")[-1].split("\\")[-1]
        if (unit["folder"] / fname).is_file():
            used.append(f"/api/media?u={uid}&f={fname}")
            return f"![{m.group(1)}](/api/media?u={uid}&f={fname})"
        return ""  # drop references to images that don't exist
    return IMG_MD_RE.sub(repl, answer), used

# ----------------------------------------------------------------------------
# 🔹 Semantic TTL cache (paper §7.3) — similar first questions reuse answers
# ----------------------------------------------------------------------------
_sem_cache = []   # [{vec (unit-normalized np), unit, qtype, ts, result}]
_sem_lock = threading.Lock()

def sem_cache_get(qvec, unit_id, qtype):
    v = np.asarray(qvec); v = v / (np.linalg.norm(v) or 1.0)
    now = time.time()
    with _sem_lock:
        _sem_cache[:] = [e for e in _sem_cache if now - e["ts"] < CACHE_TTL_SECONDS]
        for e in _sem_cache:
            if e["unit"] == unit_id and e["qtype"] == qtype and float(v @ e["vec"]) >= SEMANTIC_CACHE_SIM:
                return e["result"]
    return None

def sem_cache_put(qvec, unit_id, qtype, result):
    v = np.asarray(qvec); v = v / (np.linalg.norm(v) or 1.0)
    with _sem_lock:
        if len(_sem_cache) > 300:
            _sem_cache.pop(0)
        _sem_cache.append({"vec": v, "unit": unit_id, "qtype": qtype,
                           "ts": time.time(), "result": result})

# ----------------------------------------------------------------------------
# 🔹 FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="Wildflow")

class RouteRequest(BaseModel):
    message: str
    college: str = ""
    year: str = ""
    last_unit_id: str = ""

class AnswerRequest(BaseModel):
    message: str
    unit_ids: list
    query_type: str = "THEORY"
    history: list = []

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/meta")
def api_meta():
    colleges = {}
    for uid, u in UNITS.items():
        c = colleges.setdefault(u["college"], {"years": set(), "subjects": set(), "units": 0})
        if u["year"]:
            c["years"].add(u["year"])
        c["subjects"].add(u["subject"])
        c["units"] += 1
    order = ["First Year", "Second Year", "Third Year", "Fourth Year", "Final Year"]
    return {"colleges": [
        {"name": name,
         "years": sorted(c["years"], key=lambda y: order.index(y) if y in order else 99),
         "subjects": sorted(c["subjects"]),
         "units": c["units"]}
        for name, c in colleges.items()]}

@app.post("/api/route")
def api_route(req: RouteRequest):
    if API_KEY.startswith("PASTE_"):
        return JSONResponse(status_code=500, content={
            "error": "No Gemini API key set. Edit API_KEY in app.py or set GEMINI_API_KEY."})
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Empty message")
    college = req.college if req.college in COLLEGES else next(iter(COLLEGES))
    scope = units_in_scope(college, req.year)
    if not scope:
        return JSONResponse(status_code=404, content={"error": "No units available for this college/year."})

    subjects = sorted({u["subject"] for u in scope.values()})
    last_unit = UNITS.get(req.last_unit_id)
    last_name = f"{last_unit['subject']} — {last_unit['name']}" if last_unit else ""

    route = gemini_route(message, subjects, last_name)

    if route["subject"] == "FOLLOWUP" and last_unit and req.last_unit_id in scope:
        unit_ids, confidence = [req.last_unit_id], 1.0
        subject = last_unit["subject"]
    else:
        subject = route["subject"] if route["subject"] in subjects else None
        if subject is None:  # UNKNOWN or off-list -> local scoring across the whole scope
            subject = max(subjects, key=lambda s: max(
                (unit_topic_score(route["topic"], u)
                 for u in scope.values() if u["subject"] == s), default=0))
        unit_ids, confidence = rank_units(scope, subject, route["topic"], message)

    if not unit_ids:
        return JSONResponse(status_code=404, content={"error": "Could not route this question to any unit."})
    top = UNITS[unit_ids[0]]
    return {"subject": subject, "topic": route["topic"], "query_type": route["query_type"],
            "unit_ids": unit_ids, "unit_name": top["name"], "unit_year": top["year"],
            "confidence": round(confidence, 3)}

@app.post("/api/answer")
def api_answer(req: AnswerRequest):
    question = req.message.strip()
    unit_ids = [u for u in req.unit_ids if u in UNITS]
    if not question or not unit_ids:
        raise HTTPException(400, "Missing message or unit_ids")
    qt = req.query_type.upper()
    query_type = qt if qt in ("NUMERICAL", "GRAPHICAL") else "THEORY"

    try:
        qvec = embed_query(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    if not req.history:
        cached = sem_cache_get(qvec, unit_ids[0], query_type)
        if cached:
            return {**cached, "cached": True}

    try:
        context, topics, best_uid, sources, gallery = retrieve(unit_ids, question, qvec)
        answer = generate_answer(question, context, UNITS[best_uid], query_type, req.history)
        answer, _ = rewrite_images(answer, best_uid)

        verify = None
        if query_type == "NUMERICAL":
            verify = verify_numerical(answer)
        elif query_type == "GRAPHICAL":
            diagram_url = generate_diagram(answer)
            if diagram_url:
                answer += f"\n\n**Generated construction:**\n\n![Generated diagram]({diagram_url})"
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    unit = UNITS[best_uid]
    result = {"answer": answer, "query_type": query_type, "topics": topics,
              "gallery": gallery, "sources": sources, "verify": verify,
              "unit_id": best_uid, "unit_name": unit["name"], "subject": unit["subject"],
              "cached": False}
    if not req.history:
        sem_cache_put(qvec, unit_ids[0], query_type, result)
    return result

@app.get("/api/formulas")
def api_formulas(u: str):
    unit = UNITS.get(u)
    if not unit:
        raise HTTPException(404, "Unknown unit")
    groups = []
    for s in unit["sections"]:
        eqs = re.findall(r"\$\$.*?\$\$", s["raw"], re.DOTALL)
        if eqs:
            groups.append({"topic": s["topic"], "equations": eqs})
    return {"unit_name": unit["name"], "subject": unit["subject"], "groups": groups}

class QuizRequest(BaseModel):
    unit_id: str = ""
    topic: str = ""
    college: str = ""
    year: str = ""

@app.post("/api/quiz")
def api_quiz(req: QuizRequest):
    if API_KEY.startswith("PASTE_"):
        return JSONResponse(status_code=500, content={"error": "No Gemini API key set."})
    uid = req.unit_id if req.unit_id in UNITS else None
    if uid is None and req.topic.strip():
        college = req.college if req.college in COLLEGES else next(iter(COLLEGES))
        scope = units_in_scope(college, req.year)
        if scope:
            route = gemini_route(req.topic, sorted({u["subject"] for u in scope.values()}), "")
            subject = route["subject"] if route["subject"] in {u["subject"] for u in scope.values()} else None
            if subject is None:
                subject = max({u["subject"] for u in scope.values()},
                              key=lambda s: max((unit_topic_score(route["topic"], u)
                                                 for u in scope.values() if u["subject"] == s), default=0))
            ids, _ = rank_units(scope, subject, route["topic"], req.topic)
            uid = ids[0] if ids else None
    if uid is None:
        return JSONResponse(status_code=400, content={
            "error": "Tell me a topic to quiz you on, or ask a question first so I know your unit."})

    unit = UNITS[uid]
    sections = sorted(unit["sections"], key=lambda s: unit_topic_score(req.topic, s["topic"]) if req.topic else 0,
                      reverse=True) if req.topic else list(unit["sections"])
    material = ""
    for s in sections:
        if len(s["content"]) < 100:
            continue
        if len(material) + len(s["content"]) > 9000:
            break
        material += f"## {s['topic']}\n{s['content']}\n\n"

    prompt = f"""You are creating a practice quiz STRICTLY from the course material below
(unit "{unit['name']}", {unit['subject']}).

Generate exactly 5 exam-style questions with concise model answers:
3 conceptual, 1 application/short-numerical, 1 likely long-answer exam question.
Every question and answer must be answerable from the material alone.

Reply with ONLY a JSON array, no other text:
[{{"q": "question text", "a": "model answer (2-5 sentences, keep any equations in $$ ... $$)"}}]

COURSE MATERIAL:
{material}"""
    try:
        text = generate_with_fallback(prompt, THEORY_CHAIN)
        m = re.search(r"\[.*\]", text, re.DOTALL)
        questions = [q for q in json.loads(m.group(0)) if q.get("q") and q.get("a")][:5]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Quiz generation failed: {type(e).__name__}"})
    if not questions:
        return JSONResponse(status_code=500, content={"error": "Quiz generation returned nothing usable."})
    return {"unit_id": uid, "unit_name": unit["name"], "subject": unit["subject"], "questions": questions}

class FeedbackRequest(BaseModel):
    rating: str
    query: str = ""
    unit_id: str = ""
    unit_name: str = ""
    subject: str = ""
    query_type: str = ""
    answer_head: str = ""

@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest):
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": req.rating[:8],
             "query": req.query[:500], "unit_id": req.unit_id[:16], "unit_name": req.unit_name[:80],
             "subject": req.subject[:60], "query_type": req.query_type[:12],
             "answer_head": req.answer_head[:300]}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True}

@app.get("/api/gen/{fname}")
def api_gen(fname: str):
    if not re.fullmatch(r"[a-f0-9]{16}\.(png|jpg)", fname):
        raise HTTPException(404, "Not found")
    path = GENERATED_DIR / fname
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)

@app.get("/api/media")
def api_media(u: str, f: str):
    unit = UNITS.get(u)
    if not unit or "/" in f or "\\" in f or ".." in f:
        raise HTTPException(404, "Not found")
    path = (unit["folder"] / f).resolve()
    if not path.is_file() or path.parent != unit["folder"].resolve():
        raise HTTPException(404, "Not found")
    return FileResponse(path)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    scan_units()
    print(f"🚀 Wildflow running at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
else:
    scan_units()
