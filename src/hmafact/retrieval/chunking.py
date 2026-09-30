"""
Passage chunking for M3 CorpusBuilder.
"""
import re
from hmafact.schemas.evidence import PassageRecord

def chunk_document(doc_id: str, title: str, text: str, words: int = 120, overlap_sentences: int = 1) -> list[PassageRecord]:
    """Sentence-aware chunking with stable IDs `{source}:{doc_id}:{chunk_idx}`."""
    # Basic sentence splitter
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    chunk_idx = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = len(sentence.split())
        
        if current_word_count + sentence_words > words and current_chunk_sentences:
            # Emit current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(PassageRecord(
                passage_id=f"local_wiki:{doc_id}:{chunk_idx}",
                doc_id=doc_id,
                title=title,
                text=chunk_text,
                chunk_idx=chunk_idx,
                word_count=len(chunk_text.split()),
                dataset_origin="wikipedia"
            ))
            chunk_idx += 1
            
            # Keep overlap sentences
            overlap = min(overlap_sentences, len(current_chunk_sentences))
            if overlap > 0:
                current_chunk_sentences = current_chunk_sentences[-overlap:]
                current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
            else:
                current_chunk_sentences = []
                current_word_count = 0
                
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_words
        
    # Emit final chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append(PassageRecord(
            passage_id=f"local_wiki:{doc_id}:{chunk_idx}",
            doc_id=doc_id,
            title=title,
            text=chunk_text,
            chunk_idx=chunk_idx,
            word_count=len(chunk_text.split()),
            dataset_origin="wikipedia"
        ))
        
    return chunks
