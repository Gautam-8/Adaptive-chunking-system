"""
Base chunking strategies and factory for creating appropriate chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..classifiers.document_classifier import DocumentType, DocumentMetadata


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    start_position: int = 0
    end_position: int = 0
    
    def __post_init__(self):
        """Set end position if not provided."""
        if self.end_position == 0:
            self.end_position = self.start_position + len(self.content)


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize the chunking strategy.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, content: str, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Chunk the content based on the strategy.
        
        Args:
            content: The document content to chunk
            metadata: Document metadata from classification
            
        Returns:
            List of chunks with metadata
        """
        pass
    
    def _create_chunk(self, content: str, start_pos: int, chunk_metadata: Dict[str, Any]) -> Chunk:
        """Helper method to create a chunk with consistent metadata."""
        return Chunk(
            content=content,
            metadata=chunk_metadata,
            start_position=start_pos,
            end_position=start_pos + len(content)
        )


class SimpleChunkingStrategy(BaseChunkingStrategy):
    """Simple text chunking strategy that splits on whitespace."""
    
    def chunk(self, content: str, metadata: DocumentMetadata) -> List[Chunk]:
        """Simple chunking by character count with overlap."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            # Find the end of the current chunk
            end = min(start + self.max_chunk_size, len(content))
            
            # Try to break at a natural boundary (whitespace)
            if end < len(content):
                # Look for the last whitespace within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "strategy": "simple",
                    "document_type": metadata.document_type.value,
                    "total_chars": len(chunk_content)
                }
                
                chunks.append(self._create_chunk(chunk_content, start, chunk_metadata))
                chunk_id += 1
            
            # Move to the next chunk with overlap
            start = max(end - self.overlap, start + 1)
            
            # Prevent infinite loop
            if start >= len(content):
                break
        
        return chunks


class ChunkingStrategyFactory:
    """Factory for creating appropriate chunking strategies based on document type."""
    
    @staticmethod
    def create_strategy(document_type: DocumentType, **kwargs) -> BaseChunkingStrategy:
        """
        Create an appropriate chunking strategy for the given document type.
        
        Args:
            document_type: The type of document to chunk
            **kwargs: Additional parameters for the strategy
            
        Returns:
            An instance of the appropriate chunking strategy
        """
        # Import strategies here to avoid circular imports
        from .semantic_chunking import SemanticChunkingStrategy
        from .code_aware_chunking import CodeAwareChunkingStrategy
        from .hierarchical_chunking import HierarchicalChunkingStrategy
        
        strategy_map = {
            DocumentType.API_REFERENCE: CodeAwareChunkingStrategy,
            DocumentType.CODE_DOCUMENTATION: CodeAwareChunkingStrategy,
            DocumentType.TECHNICAL_DOC: HierarchicalChunkingStrategy,
            DocumentType.POLICY_DOCUMENT: HierarchicalChunkingStrategy,
            DocumentType.TUTORIAL: HierarchicalChunkingStrategy,
            DocumentType.SUPPORT_TICKET: SemanticChunkingStrategy,
            DocumentType.TROUBLESHOOTING: SemanticChunkingStrategy,
            DocumentType.UNKNOWN: SimpleChunkingStrategy,
        }
        
        strategy_class = strategy_map.get(document_type, SimpleChunkingStrategy)
        return strategy_class(**kwargs)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available chunking strategies."""
        return [
            "SimpleChunkingStrategy",
            "SemanticChunkingStrategy", 
            "CodeAwareChunkingStrategy",
            "HierarchicalChunkingStrategy"
        ] 