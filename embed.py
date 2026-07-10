# ============================================================================
#  WILDFLOW — EMBEDDER (admin tool, run manually)
# ============================================================================
#  Point UNIT_FOLDERS at one or more unit folders. Each unit folder must
#  contain exactly one .md file (the authoritative Markdown for that unit).
#  A fresh Qdrant database is (re)built INSIDE the unit folder as "qdrant_db".
#
#  Usage:
#      python embed.py                          -> processes UNIT_FOLDERS below
#      python embed.py "D:\path\to\Unit 1" ...  -> processes the given folders
#
#  Vector recipe (unchanged from the original design):
#      multilingual-e5-large (1024D) + CLIP ViT-B/32 text (512D) = 1536D cosine
#  Only cleaned text is embedded; $$KaTeX$$ blocks and image references are
#  stripped before embedding but remain in the Markdown (the source of truth).
# ============================================================================

import os
os.environ.setdefault("USE_TF", "0")   # skip transformers' TensorFlow backend (Keras 3 conflict)

import re
import sys
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ----------------------------------------------------------------------------
# 🔹 CONFIGURATION — edit these paths, then run
# ----------------------------------------------------------------------------
UNIT_FOLDERS = [
    # r"C:\Users\rohan\Desktop\LLMstudy\MITwpuEmbeddedMD\B.tech\Engineering Chemistry\First Year\Unit 1_Speciality Polymers (Engineering Chemistry)",
]

COLLECTION_NAME = "text_embeddings"   # keep this name; the app auto-detects it
QDRANT_DIR_NAME = "qdrant_db"         # created inside each unit folder
TEXT_MODEL_NAME = "intfloat/multilingual-e5-large"   # 1024D
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"     # 512D
EMBEDDING_DIM   = 1536
BATCH_SIZE      = 8

# ----------------------------------------------------------------------------
# 🔹 Markdown section parser
#    Splits on EVERY heading level (## through ######) because the Surya/Marker
#    output uses ###, ####, ##### and ###### for topics — not just ##.
# ----------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{2,6})\s*(.+?)\s*$", re.MULTILINE)

def clean_for_embedding(text):
    """Remove KaTeX blocks and image references; collapse blank lines."""
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def parse_sections(md_content):
    """Return [{topic, level, raw, content}] — raw keeps equations/images,
    content is the cleaned text that gets embedded."""
    sections = []
    matches = list(HEADING_RE.finditer(md_content))

    # Content before the first heading
    pre_end = matches[0].start() if matches else len(md_content)
    pre_clean = clean_for_embedding(md_content[:pre_end])
    if pre_clean:
        sections.append({"topic": "Introduction", "level": 2,
                         "raw": md_content[:pre_end].strip(), "content": pre_clean})

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_content)
        body = md_content[m.end():end]
        content = clean_for_embedding(body)
        if not content:
            continue
        sections.append({
            "topic": m.group(2).lstrip("#").strip(),
            "level": len(m.group(1)),
            "raw": md_content[m.start():end].strip(),
            "content": content,
        })
    return sections

# ----------------------------------------------------------------------------
# 🔹 Embedding (identical recipe to the existing databases)
# ----------------------------------------------------------------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading embedding models on {device} ...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device,
                                     model_kwargs={"low_cpu_mem_usage": True})
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, low_cpu_mem_usage=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("✅ Models loaded.")
    return device, text_model, clip_model, clip_processor

def embed_batch(texts, device, text_model, clip_model, clip_processor):
    """1024D sentence embedding + 512D CLIP text embedding per text."""
    sent = text_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    clip_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs).cpu().numpy()
        clip_vecs.extend(feats)
    return [list(map(float, s)) + list(map(float, c)) for s, c in zip(sent, clip_vecs)]

# ----------------------------------------------------------------------------
# 🔹 Per-unit processing: parse -> embed -> rebuild collection
# ----------------------------------------------------------------------------
def process_unit(folder, device, text_model, clip_model, clip_processor):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"❌ Not a folder, skipping: {folder}")
        return

    md_files = [f for f in os.listdir(folder) if f.lower().endswith(".md")]
    if len(md_files) != 1:
        print(f"❌ Expected exactly one .md in {folder} (found {len(md_files)}), skipping.")
        return
    md_path = os.path.join(folder, md_files[0])

    print(f"\n📄 {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        sections = parse_sections(f.read())
    if not sections:
        print("⚠️  No sections with text content found, skipping.")
        return
    print(f"   Parsed {len(sections)} sections.")

    vectors = embed_batch([s["content"] for s in sections],
                          device, text_model, clip_model, clip_processor)

    db_path = os.path.join(folder, QDRANT_DIR_NAME)
    client = QdrantClient(path=db_path)
    try:
        # Full rebuild every run: no stale chunks can survive an edited .md
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i + 1, vector=vec, payload={
                "type": "text",
                "topic": s["topic"],
                "content": s["content"],
                "source_file": md_files[0],
            })
            for i, (s, vec) in enumerate(zip(sections, vectors))
        ]
        for i in range(0, len(points), 64):
            client.upsert(COLLECTION_NAME, points=points[i:i + 64], wait=True)
        print(f"✅ Stored {len(points)} chunks -> {db_path} [{COLLECTION_NAME}]")
    finally:
        client.close()   # release the folder lock so the app can open it

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    folders = sys.argv[1:] or UNIT_FOLDERS
    if not folders:
        print("Nothing to do. Add paths to UNIT_FOLDERS at the top of this file,")
        print('or run:  python embed.py "D:\\path\\to\\Unit 1" "D:\\path\\to\\Unit 2"')
        sys.exit(1)

    device, text_model, clip_model, clip_processor = load_models()
    for folder in folders:
        process_unit(folder, device, text_model, clip_model, clip_processor)
    print("\n🏁 Done.")
