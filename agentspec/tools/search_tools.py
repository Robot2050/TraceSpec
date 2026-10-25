"""BrowseComp-plus retrieval tools: sqlite FTS5 index over the corpus.

The corpus (~100k documents, ~3GB) is indexed once into a disk-backed FTS5
table with built-in BM25 ranking; every evaluation process then opens
read-only connections. This avoids holding a tokenized corpus in RAM.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    docid UNINDEXED,
    text,
    tokenize = 'unicode61'
);
"""

_TERM_RE = re.compile(r"[A-Za-z0-9_]+")


def index_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "browsecomp_fts.db"


def ensure_index(corpus_path: str | Path, index_dir: str | Path) -> Path:
    """Build the FTS index if missing; returns the db path. Call this once
    from the main process before forking workers."""
    db_path = index_path(index_dir)
    if db_path.exists():
        return db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(corpus_path)
    tmp_path = db_path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[agentspec] building FTS index for {corpus_path} -> {db_path}", flush=True)
    t0 = time.time()
    con = sqlite3.connect(tmp_path)
    con.executescript(_SCHEMA)
    batch = []
    n_docs = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch.append((str(row["docid"]), row["text"]))
            if len(batch) >= 2000:
                con.executemany("INSERT INTO docs(docid, text) VALUES (?, ?)", batch)
                con.commit()
                n_docs += len(batch)
                batch = []
                print(f"[agentspec]   indexed {n_docs} docs ({time.time() - t0:.0f}s)", flush=True)
    if batch:
        con.executemany("INSERT INTO docs(docid, text) VALUES (?, ?)", batch)
        con.commit()
        n_docs += len(batch)
    con.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    con.commit()
    con.close()
    tmp_path.rename(db_path)
    print(f"[agentspec] index ready: {n_docs} docs in {time.time() - t0:.0f}s", flush=True)
    return db_path


def _sanitize_query(query: str, max_terms: int = 16) -> str:
    terms = _TERM_RE.findall(query)[:max_terms]
    # OR-join so recall dominates; FTS5's bm25() handles ranking.
    return " OR ".join(f'"{t}"' for t in terms)


class SearchTool:
    """Per-process read-only search/read access to the indexed corpus."""

    def __init__(self, db_path: str | Path, *, top_k: int = 5, max_chars: int = 4000):
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.top_k = top_k
        self.max_chars = max_chars

    def search(self, query: str, *, exclude: set[str] | None = None) -> str:
        """Top-k BM25 snippets for the query. Docids in `exclude` (e.g.
        documents the agent has already read in full) are filtered out so the
        observation only carries fresh material; a few extra candidates are
        fetched to backfill filtered hits."""
        fts_query = _sanitize_query(query)
        if not fts_query:
            return "[empty query]"
        try:
            rows = self.con.execute(
                "SELECT docid, snippet(docs, 1, '[', ']', ' … ', 24) AS snip "
                "FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                (fts_query, self.top_k + 10 if exclude else self.top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            return f"[search error: {exc}]"
        if not rows:
            return "[no results]"
        if exclude:
            rows = [row for row in rows if row[0] not in exclude]
            if not rows:
                return "[no new results: every top hit was already read]"
        rows = rows[: self.top_k]
        parts = [f"(docid={docid}) {snip}" for docid, snip in rows]
        text = "\n\n".join(parts)
        return text[: self.max_chars]

    def read(self, docid: str) -> str:
        row = self.con.execute(
            "SELECT text FROM docs WHERE docid = ?", (str(docid),)
        ).fetchone()
        if row is None:
            return f"[document {docid} not found]"
        text = row[0]
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text

    def close(self) -> None:
        self.con.close()
