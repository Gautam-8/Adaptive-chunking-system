"""
Main pipeline orchestrator for the adaptive chunking system.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

from ..classifiers.document_classifier import DocumentClassifier, DocumentMetadata
from ..strategies.chunking_strategies import ChunkingStrategyFactory, Chunk


@dataclass
class ProcessingResult:
    """Result of processing a document through the adaptive chunking pipeline."""
    
    document_id: str
    source_path: Optional[str]
    document_metadata: Optional[DocumentMetadata]
    chunks: List[Chunk]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = asdict(self)
        # Convert chunks to dictionaries
        result["chunks"] = [asdict(chunk) for chunk in self.chunks]
        # Convert document metadata
        if self.document_metadata:
            result["document_metadata"] = asdict(self.document_metadata)
            result["document_metadata"]["document_type"] = self.document_metadata.document_type.value
        else:
            result["document_metadata"] = None
        return result


class AdaptiveChunkingPipeline:
    """
    Main pipeline for adaptive document chunking.
    
    This pipeline:
    1. Classifies documents to determine their type and structure
    2. Selects appropriate chunking strategy based on classification
    3. Processes documents into optimally sized chunks
    4. Provides metadata and performance tracking
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 overlap: int = 100,
                 enable_logging: bool = True):
        """
        Initialize the adaptive chunking pipeline.
        
        Args:
            max_chunk_size: Default maximum chunk size in characters
            overlap: Default overlap between chunks in characters
            enable_logging: Whether to enable logging
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.classifier = DocumentClassifier()
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "avg_chunks_per_document": 0.0,
            "avg_processing_time": 0.0
        }
    
    def process_document(self, 
                        content: str,
                        document_id: str,
                        source_path: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        strategy_params: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process a single document through the adaptive chunking pipeline.
        
        Args:
            content: The document content as a string
            document_id: Unique identifier for the document
            source_path: Optional path to source file
            metadata: Optional additional metadata
            strategy_params: Optional parameters for chunking strategy
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing document: {document_id}")
            
            # Step 1: Classify document
            doc_metadata = self.classifier.classify(content, metadata)
            self.logger.info(f"Document classified as: {doc_metadata.document_type.value} "
                           f"(confidence: {doc_metadata.confidence:.2f})")
            
            # Step 2: Select and configure chunking strategy
            strategy_config = strategy_params or {}
            strategy_config.setdefault("max_chunk_size", self.max_chunk_size)
            strategy_config.setdefault("overlap", self.overlap)
            
            strategy = ChunkingStrategyFactory.create_strategy(
                doc_metadata.document_type, 
                **strategy_config
            )
            
            # Step 3: Chunk the document
            chunks = strategy.chunk(content, doc_metadata)
            self.logger.info(f"Document chunked into {len(chunks)} chunks")
            
            # Step 4: Create result
            processing_time = time.time() - start_time
            result = ProcessingResult(
                document_id=document_id,
                source_path=source_path,
                document_metadata=doc_metadata,
                chunks=chunks,
                processing_time=processing_time,
                success=True
            )
            
            # Update stats
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {document_id}: {str(e)}")
            
            result = ProcessingResult(
                document_id=document_id,
                source_path=source_path,
                document_metadata=None,
                chunks=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
            
            self._update_stats(result)
            return result
    
    def process_documents(self, 
                         documents: List[Dict[str, Any]],
                         batch_size: int = 10) -> List[ProcessingResult]:
        """
        Process multiple documents in batches.
        
        Args:
            documents: List of document dictionaries with 'content', 'id', and optional metadata
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} "
                           f"({len(batch)} documents)")
            
            for doc in batch:
                result = self.process_document(
                    content=doc["content"],
                    document_id=doc["id"],
                    source_path=doc.get("source_path"),
                    metadata=doc.get("metadata"),
                    strategy_params=doc.get("strategy_params")
                )
                results.append(result)
        
        return results
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    document_id: Optional[str] = None) -> ProcessingResult:
        """
        Process a document from a file.
        
        Args:
            file_path: Path to the file to process
            document_id: Optional document ID (uses filename if not provided)
            
        Returns:
            ProcessingResult
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Use filename as document ID if not provided
        if document_id is None:
            document_id = file_path.stem
        
        # Extract metadata from file
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": file_path.stat().st_size
        }
        
        return self.process_document(
            content=content,
            document_id=document_id,
            source_path=str(file_path),
            metadata=metadata
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "avg_chunks_per_document": 0.0,
            "avg_processing_time": 0.0
        }
    
    def _update_stats(self, result: ProcessingResult):
        """Update processing statistics with new result."""
        self.processing_stats["total_documents"] += 1
        
        if result.success:
            self.processing_stats["successful_documents"] += 1
            self.processing_stats["total_chunks"] += len(result.chunks)
        else:
            self.processing_stats["failed_documents"] += 1
        
        # Update averages
        total_docs = self.processing_stats["total_documents"]
        if total_docs > 0:
            self.processing_stats["avg_chunks_per_document"] = (
                self.processing_stats["total_chunks"] / 
                self.processing_stats["successful_documents"]
                if self.processing_stats["successful_documents"] > 0 else 0.0
            )
            
            # Simple running average for processing time
            current_avg = self.processing_stats["avg_processing_time"]
            self.processing_stats["avg_processing_time"] = (
                (current_avg * (total_docs - 1) + result.processing_time) / total_docs
            ) 