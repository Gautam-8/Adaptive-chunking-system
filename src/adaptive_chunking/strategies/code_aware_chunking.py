"""
Code-aware chunking strategy that preserves code blocks and API structure.
"""

import re
from typing import List, Dict, Any, Tuple

from .chunking_strategies import BaseChunkingStrategy, Chunk
from ..classifiers.document_classifier import DocumentMetadata


class CodeAwareChunkingStrategy(BaseChunkingStrategy):
    """
    Code-aware chunking strategy that preserves code blocks and API structure.
    
    This strategy:
    - Keeps code blocks intact
    - Preserves API endpoint definitions
    - Maintains code-documentation relationships
    - Handles different programming languages
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 150):
        """Initialize code-aware chunking strategy with larger default sizes."""
        super().__init__(max_chunk_size, overlap)
    
    def chunk(self, content: str, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Chunk content while preserving code blocks and API structure.
        
        Args:
            content: The document content to chunk
            metadata: Document metadata from classification
            
        Returns:
            List of code-aware chunks
        """
        chunks = []
        
        # Find all code blocks first
        code_blocks = self._find_code_blocks(content)
        
        # Split content into sections around code blocks
        sections = self._split_around_code_blocks(content, code_blocks)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for section in sections:
            section_content, section_type = section
            
            # If this is a code block, try to keep it intact
            if section_type == "code":
                # If adding this code block would exceed max size, finalize current chunk
                if current_chunk and len(current_chunk) + len(section_content) > self.max_chunk_size:
                    if current_chunk.strip():
                        chunk_metadata = {
                            "chunk_id": chunk_id,
                            "strategy": "code_aware",
                            "document_type": metadata.document_type.value,
                            "total_chars": len(current_chunk.strip()),
                            "contains_code": self._contains_code(current_chunk),
                            "code_language": self._detect_code_language(current_chunk)
                        }
                        
                        chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
                        chunk_id += 1
                    
                    # Start new chunk with the code block
                    current_chunk = section_content
                    current_start = self._calculate_start_position(content, section_content)
                else:
                    # Add code block to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + section_content
                    else:
                        current_chunk = section_content
            
            else:  # Regular text section
                # Split large text sections if needed
                if len(section_content) > self.max_chunk_size:
                    text_chunks = self._split_text_section(section_content)
                    for text_chunk in text_chunks:
                        if current_chunk and len(current_chunk) + len(text_chunk) > self.max_chunk_size:
                            # Finalize current chunk
                            if current_chunk.strip():
                                chunk_metadata = {
                                    "chunk_id": chunk_id,
                                    "strategy": "code_aware",
                                    "document_type": metadata.document_type.value,
                                    "total_chars": len(current_chunk.strip()),
                                    "contains_code": self._contains_code(current_chunk),
                                    "code_language": self._detect_code_language(current_chunk)
                                }
                                
                                chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
                                chunk_id += 1
                            
                            current_chunk = text_chunk
                            current_start = self._calculate_start_position(content, text_chunk)
                        else:
                            if current_chunk:
                                current_chunk += "\n\n" + text_chunk
                            else:
                                current_chunk = text_chunk
                else:
                    # Add entire text section
                    if current_chunk and len(current_chunk) + len(section_content) > self.max_chunk_size:
                        # Finalize current chunk
                        if current_chunk.strip():
                            chunk_metadata = {
                                "chunk_id": chunk_id,
                                "strategy": "code_aware",
                                "document_type": metadata.document_type.value,
                                "total_chars": len(current_chunk.strip()),
                                "contains_code": self._contains_code(current_chunk),
                                "code_language": self._detect_code_language(current_chunk)
                            }
                            
                            chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
                            chunk_id += 1
                        
                        current_chunk = section_content
                        current_start = self._calculate_start_position(content, section_content)
                    else:
                        if current_chunk:
                            current_chunk += "\n\n" + section_content
                        else:
                            current_chunk = section_content
        
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_metadata = {
                "chunk_id": chunk_id,
                "strategy": "code_aware",
                "document_type": metadata.document_type.value,
                "total_chars": len(current_chunk.strip()),
                "contains_code": self._contains_code(current_chunk),
                "code_language": self._detect_code_language(current_chunk)
            }
            
            chunks.append(self._create_chunk(current_chunk.strip(), current_start, chunk_metadata))
        
        return chunks
    
    def _find_code_blocks(self, content: str) -> List[Tuple[int, int, str]]:
        """Find all code blocks in the content."""
        code_blocks = []
        
        # Find fenced code blocks (```)
        pattern = r'```[\s\S]*?```'
        for match in re.finditer(pattern, content):
            code_blocks.append((match.start(), match.end(), match.group()))
        
        # Find inline code blocks (single backticks) - only if they're substantial
        pattern = r'`[^`\n]{20,}`'  # At least 20 characters
        for match in re.finditer(pattern, content):
            code_blocks.append((match.start(), match.end(), match.group()))
        
        return sorted(code_blocks, key=lambda x: x[0])
    
    def _split_around_code_blocks(self, content: str, code_blocks: List[Tuple[int, int, str]]) -> List[Tuple[str, str]]:
        """Split content into sections around code blocks."""
        sections = []
        last_end = 0
        
        for start, end, code_content in code_blocks:
            # Add text before code block
            if start > last_end:
                text_content = content[last_end:start].strip()
                if text_content:
                    sections.append((text_content, "text"))
            
            # Add code block
            sections.append((code_content, "code"))
            last_end = end
        
        # Add remaining text
        if last_end < len(content):
            text_content = content[last_end:].strip()
            if text_content:
                sections.append((text_content, "text"))
        
        return sections
    
    def _split_text_section(self, text: str) -> List[str]:
        """Split large text sections into smaller chunks."""
        # Try to split by API endpoints or headers first
        api_pattern = r'(^#{1,6}.*$|^[A-Z]+ /.*$)'
        parts = re.split(api_pattern, text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if not part or not part.strip():
                continue
                
            if len(current_chunk) + len(part) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return bool(re.search(r'```[\s\S]*?```|`[^`\n]+`', text))
    
    def _detect_code_language(self, text: str) -> str:
        """Detect the programming language in code blocks."""
        # Look for language specifiers in fenced code blocks
        pattern = r'```(\w+)'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        
        # Simple heuristics for common languages
        if re.search(r'\bdef\b|\bclass\b|\bimport\b', text):
            return "python"
        elif re.search(r'\bfunction\b|\bvar\b|\bconst\b|\blet\b', text):
            return "javascript"
        elif re.search(r'\bpublic\b|\bprivate\b|\bclass\b.*{', text):
            return "java"
        elif re.search(r'GET|POST|PUT|DELETE', text):
            return "http"
        
        return "unknown"
    
    def _calculate_start_position(self, full_content: str, chunk_content: str) -> int:
        """Calculate the start position of chunk in full content."""
        pos = full_content.find(chunk_content[:50])
        return max(0, pos) 