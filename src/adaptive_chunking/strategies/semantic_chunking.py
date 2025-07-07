"""
Semantic chunking strategy that preserves meaning and context.
"""

import re
from typing import List, Dict, Any

from .chunking_strategies import BaseChunkingStrategy, Chunk
from ..classifiers.document_classifier import DocumentMetadata


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Semantic chunking strategy that tries to preserve semantic meaning.
    
    This strategy attempts to chunk at natural semantic boundaries like:
    - Sentence endings
    - Paragraph breaks
    - Section breaks
    - Topic changes
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        """Initialize semantic chunking strategy."""
        super().__init__(max_chunk_size, overlap)
    
    def chunk(self, content: str, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Chunk content at semantic boundaries.
        
        Args:
            content: The document content to chunk
            metadata: Document metadata from classification
            
        Returns:
            List of semantically meaningful chunks
        """
        chunks = []
        
        # First, try to split by paragraphs
        paragraphs = self._split_by_paragraphs(content)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk.strip():
                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "strategy": "semantic",
                        "document_type": metadata.document_type.value,
                        "total_chars": len(current_chunk.strip()),
                        "semantic_type": "paragraph_group"
                    }
                    
                    chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
                    chunk_id += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap(chunks[-1].content if chunks else "", paragraph)
                current_start = self._calculate_start_position(content, current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_metadata = {
                "chunk_id": chunk_id,
                "strategy": "semantic",
                "document_type": metadata.document_type.value,
                "total_chars": len(current_chunk.strip()),
                "semantic_type": "paragraph_group"
            }
            
            chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
        
        return chunks
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If paragraphs are too large, split by sentences
        refined_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > self.max_chunk_size:
                sentences = self._split_by_sentences(paragraph)
                refined_paragraphs.extend(sentences)
            else:
                refined_paragraphs.append(paragraph)
        
        return refined_paragraphs
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap(self, previous_chunk: str, current_paragraph: str) -> str:
        """Get overlap content from previous chunk."""
        if not previous_chunk or self.overlap == 0:
            return current_paragraph
        
        # Take last few sentences from previous chunk as overlap
        overlap_text = previous_chunk[-self.overlap:] if len(previous_chunk) > self.overlap else previous_chunk
        
        # Try to find sentence boundary for clean overlap
        last_sentence_end = overlap_text.rfind('.')
        if last_sentence_end > 0:
            overlap_text = overlap_text[last_sentence_end + 1:].strip()
        
        return overlap_text + "\n\n" + current_paragraph if overlap_text else current_paragraph
    
    def _calculate_start_position(self, full_content: str, chunk_content: str) -> int:
        """Calculate the start position of chunk in full content."""
        # Simple approach: find first occurrence
        # In a real implementation, you'd want more sophisticated position tracking
        pos = full_content.find(chunk_content[:50])  # Use first 50 chars to find position
        return max(0, pos) 