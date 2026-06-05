from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from collections import Counter
import textwrap
import re
import glob 
import os
from config import RAW_DIR, MAX_CHUNK_CHARS


@dataclass
class Chunk:
    """
    A single text chunk with provenance metadata.

    Attributes
    ----------
    chunk_id  : sequential integer assigned during chunking
    text      : the actual text content of this chunk
    strategy  : which chunking strategy produced this chunk
    section   : top-level section heading this chunk belongs to
    level     : structural depth (1=section, 2=paragraph, 3=rule, 4=subitem)
    parent    : text of the immediate parent element, if any
    """
    chunk_id : int
    text     : str
    source  : str  = ""
    section    : str  = ""
    subsection   : str  = ""

    def preview(self, width: int = 80) -> str:
        """Return a compact, wrapped display of this chunk."""
        header = (
            f"[{self.chunk_id}] source={self.source}  "
            f"section={self.section}  subsection='{self.subsection}'"
        )
        body = textwrap.fill(self.text, width=width)
        return f"{header}\n{body}"


def _overflow_split(
    body: str, source: str, section: str, subsection: str, start_idx: int,
) -> list[Chunk]:
    """
    Break an oversized body on sentence boundaries.
    Sentences are greedily packed until MAX_CHUNK_CHARS is reached.
    """
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    chunks, buf, idx = [], [], start_idx

    for sent in sentences:
        if buf and sum(len(s) for s in buf) + len(sent) + 1 > MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id = idx,
                text= " ".join(buf),
                source=source,
                section=section,
                subsection=subsection
            )
            )
            idx += 1
            buf = []
        buf.append(sent)

    if buf:
        chunks.append(Chunk(
            chunk_id = idx,
            text= " ".join(buf),
            source=source,
            section=section,
            subsection=subsection
        ))
    return chunks

def _chunk_pages(text: str, source: str) -> List[Chunk]:
    """Main chunking section, to chunk through a wikipedia article
    I will take the approach of:
    1. Force flushing whenever a Major Header is encountered, identified as ===== from when we build our text files scraped from wikipedia
    2. Updates the subsection attribute only but does not force a flush
    3. Prose is flushed when the buffer exceeds our MAX_CHUNK_CHARS
    """

    lines = text.splitlines()
    chunks = []
    section, subsection, buf, idx = "Overview", "Overview", [], 0

    def flush():
        nonlocal idx 
        body = " ".join(buf).strip()
        buf.clear()
        if not body:
            return
        if len(body) > MAX_CHUNK_CHARS:
            overflow = _overflow_split(body, source, section, subsection, idx)
            chunks.extend(overflow)
            idx += len(overflow)
        else:
            chunks.append(Chunk(
                chunk_id = idx,
                text=body,
                source=source,
                section=section,
                subsection=subsection
            ))
            idx+= 1
    
    for line in lines:
        s = line.strip()
        if s.startswith("====="):
            flush()
            section = s.replace('=','').strip()
            subsection = "Overview"
        elif s.startswith('=='):
            subsection=s.replace('=','').strip()
        else:
            buf.append(s)
    flush()
    return [c for c in chunks if c.text]
        
def load_and_chunk(docs_dir: str) -> list[dict]:
    """
    Load every .txt file in docs_dir, apply the chunker
    Returns a flat list of chunk dicts:
      {text, source, group, section, subsection, chunk_idx}
    """
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in '{docs_dir}'.")

    all_chunks = []
    for path in paths:
        text   = Path(path).read_text(encoding="utf-8")
        source = Path(path).name
        doc_chunks = _chunk_pages(text, source)

        all_chunks.extend(doc_chunks)
        print(f"{source}  ->  {len(doc_chunks)} chunks")

    print(f"\n[load] {len(paths)} files -> {len(all_chunks)} total chunks.")
    return all_chunks

def print_chunks(chunks: List[Chunk], n: int = 5) -> None:
    """Pretty-print the first n chunks."""
    border = "═" * 80
    for c in chunks[:n]:
        print(border)
        print(c.preview())
    print(border)

