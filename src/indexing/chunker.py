"""
Document chunking strategies.

Each chunking strategy produces chunks that preserve character offsets back to
the original document. This is critical for evaluation against the LegalBench-RAG
benchmark which uses character-level precision and recall.
"""

import re
from dataclasses import dataclass
from typing import List

from src.config import ChunkingConfig


@dataclass
class Chunk:
    """A chunk of text from a document with its position metadata."""
    text: str
    file_path: str
    start: int          # character offset in original document
    end: int            # character offset in original document
    chunk_id: int       # sequential index within the document

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_document_fixed(
    text: str,
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 128,
    min_chunk_size: int = 50,
) -> List[Chunk]:
    """
    Split a document into fixed-size chunks with overlap.
    
    Chunks are aligned to word boundaries where possible to avoid splitting
    words. Each chunk records its exact character offsets in the source document.
    """
    chunks = []
    text_len = len(text)

    if text_len == 0:
        return chunks

    start = 0
    chunk_id = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to align to a word boundary (look for whitespace near the end)
        if end < text_len:
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary + 1  # include the space

        chunk_text = text[start:end]

        if len(chunk_text.strip()) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text,
                file_path=file_path,
                start=start,
                end=end,
                chunk_id=chunk_id,
            ))
            chunk_id += 1

        # Advance by (chunk_size - overlap), but not past end
        step = max(chunk_size - overlap, 1)
        start += step

    return chunks


def chunk_document_sentence(
    text: str,
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 128,
    min_chunk_size: int = 50,
) -> List[Chunk]:
    """
    Split a document into chunks at sentence boundaries.
    
    Sentences are detected by common legal document patterns (periods followed
    by spaces and capitals, semicolons at clause boundaries, etc.). Chunks
    accumulate sentences until they approach chunk_size, then start a new chunk
    with overlap from the previous sentences.
    """
    # Split on sentence boundaries common in legal text
    sentence_pattern = re.compile(
        r'(?<=[.;!?])\s+(?=[A-Z"(\d])|(?<=\n)\s*(?=\S)'
    )

    sentences = []
    last_end = 0
    for match in sentence_pattern.finditer(text):
        split_pos = match.start()
        if split_pos > last_end:
            sentences.append((last_end, split_pos + len(match.group())))
        last_end = split_pos + len(match.group())

    if last_end < len(text):
        sentences.append((last_end, len(text)))

    if not sentences:
        return chunk_document_fixed(text, file_path, chunk_size, overlap, min_chunk_size)

    chunks = []
    chunk_id = 0
    current_start_idx = 0

    while current_start_idx < len(sentences):
        # Accumulate sentences until we approach chunk_size
        current_start = sentences[current_start_idx][0]
        current_end = current_start
        sent_idx = current_start_idx

        while sent_idx < len(sentences):
            sent_end = sentences[sent_idx][1]
            if sent_end - current_start > chunk_size and sent_idx > current_start_idx:
                break
            current_end = sent_end
            sent_idx += 1

        chunk_text = text[current_start:current_end]

        if len(chunk_text.strip()) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text,
                file_path=file_path,
                start=current_start,
                end=current_end,
                chunk_id=chunk_id,
            ))
            chunk_id += 1

        # Calculate how many sentences to skip for overlap
        overlap_chars = 0
        overlap_start_idx = sent_idx
        for rev_idx in range(sent_idx - 1, current_start_idx, -1):
            s_start, s_end = sentences[rev_idx]
            overlap_chars += s_end - s_start
            if overlap_chars >= overlap:
                overlap_start_idx = rev_idx
                break

        if overlap_start_idx <= current_start_idx:
            current_start_idx = sent_idx
        else:
            current_start_idx = overlap_start_idx

        if current_start_idx >= len(sentences):
            break

    return chunks


def chunk_document(
    text: str,
    file_path: str,
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Chunk a document according to the given configuration.
    
    Routes to the appropriate chunking strategy based on config.strategy.
    """
    if config.strategy == "sentence":
        return chunk_document_sentence(
            text, file_path,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
        )
    else:  # "fixed" is the default
        return chunk_document_fixed(
            text, file_path,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
        )

