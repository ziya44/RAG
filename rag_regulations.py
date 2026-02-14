from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents(source_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            docs.append((str(path), read_text_file(path)))
        elif suffix == ".pdf":
            docs.append((str(path), read_pdf_file(path)))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    clean = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = end - overlap
    return chunks


def build_chunks(source_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_documents(source_dir):
        split = chunk_text(text)
        for idx, part in enumerate(split):
            if part.strip():
                chunks.append(Chunk(text=part, source=source, chunk_id=idx))
    return chunks


def save_index(index_dir: Path, vectors: np.ndarray, chunks: list[Chunk]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "regulations.faiss"))

    metadata = [asdict(chunk) for chunk in chunks]
    (index_dir / "chunks.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(str(index_dir / "regulations.faiss"))
    meta_raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in meta_raw]
    return index, chunks


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    arr = model.encode(list(texts), convert_to_numpy=True)
    return arr.astype(np.float32)


def ingest(source_dir: Path, index_dir: Path, embedding_model: str) -> None:
    chunks = build_chunks(source_dir)
    if not chunks:
        raise ValueError("Kaynak klasörde okunabilir dosya bulunamadı.")

    model = SentenceTransformer(embedding_model)
    vectors = embed_texts(model, (c.text for c in chunks))
    save_index(index_dir, vectors, chunks)
    print(f"İndeks oluşturuldu. Toplam chunk: {len(chunks)}")


def retrieve(
    question: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks: list[Chunk],
    top_k: int = 5,
    min_score: float = 0.35,
) -> list[tuple[float, Chunk]]:
    q = embed_texts(embedder, [question])
    faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)

    result: list[tuple[float, Chunk]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        if score >= min_score:
            result.append((float(score), chunks[idx]))
    return result


def build_prompt(question: str, contexts: list[tuple[float, Chunk]]) -> str:
    sources = []
    for i, (score, chunk) in enumerate(contexts, start=1):
        sources.append(
            f"[{i}] Kaynak: {chunk.source} | Parça: {chunk.chunk_id} | Skor: {score:.3f}\n{chunk.text}"
        )

    context_block = "\n\n".join(sources)
    return f"""
Sen eğitim yönetmelikleri asistanısın.
Sadece aşağıdaki kaynak metinlere dayanarak cevap ver.
Kaynaklarda açıkça geçmeyen bilgi için tahmin yürütme.
Eğer cevap kaynaklarda yoksa şu şekilde yanıtla: "Bu sorunun cevabı yüklenen yönetmeliklerde bulunamadı."

Soru:
{question}

Kaynaklar:
{context_block}

Yanıt formatı:
1) Kısa cevap
2) Dayanak maddeler/alıntılar
3) Kaynak referansları ([1], [2] gibi)
""".strip()


def answer_question(
    index_dir: Path,
    question: str,
    embedding_model: str,
    llm_model: str,
    top_k: int,
    min_score: float,
) -> None:
    index, chunks = load_index(index_dir)
    embedder = SentenceTransformer(embedding_model)
    contexts = retrieve(
        question=question,
        embedder=embedder,
        index=index,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )

    if not contexts:
        print("Bu sorunun cevabı yüklenen yönetmeliklerde bulunamadı.")
        return

    prompt = build_prompt(question, contexts)

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
    )

    output = generator(prompt)[0]["generated_text"]
    answer = output[len(prompt) :].strip()

    print("\n=== CEVAP ===\n")
    print(answer)
    print("\n=== KULLANILAN KAYNAKLAR ===")
    for i, (score, chunk) in enumerate(contexts, start=1):
        print(f"[{i}] {chunk.source} | parça={chunk.chunk_id} | skor={score:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eğitim yönetmelikleri için RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Dökümanlardan indeks oluştur")
    ingest_p.add_argument("--source-dir", type=Path, required=True)
    ingest_p.add_argument("--index-dir", type=Path, required=True)
    ingest_p.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    ask_p = sub.add_parser("ask", help="İndeksten soru cevapla")
    ask_p.add_argument("--index-dir", type=Path, required=True)
    ask_p.add_argument("--question", required=True)
    ask_p.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ask_p.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ask_p.add_argument("--top-k", type=int, default=5)
    ask_p.add_argument("--min-score", type=float, default=0.35)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(
            source_dir=args.source_dir,
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
        )
    elif args.command == "ask":
        answer_question(
            index_dir=args.index_dir,
            question=args.question,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            min_score=args.min_score,
        )


if __name__ == "__main__":
    main()
