"""
Hierarchical chunking strategy that preserves document structure and hierarchy.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

from .chunking_strategies import BaseChunkingStrategy, Chunk
from ..classifiers.document_classifier import DocumentMetadata


class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    """
    Hierarchical chunking strategy that preserves document structure.
    
    This strategy:
    - Respects heading hierarchy (H1, H2, H3, etc.)
    - Keeps related sections together
    - Preserves numbered lists and procedures
    - Maintains parent-child relationships
    """
    
    def __init__(self, max_chunk_size: int = 1200, overlap: int = 100):
        """Initialize hierarchical chunking strategy."""
        super().__init__(max_chunk_size, overlap)
    
    def chunk(self, content: str, metadata: DocumentMetadata) -> List[Chunk]:
        """
        Chunk content while preserving hierarchical structure.
        
        Args:
            content: The document content to chunk
            metadata: Document metadata from classification
            
        Returns:
            List of hierarchically structured chunks
        """
        chunks = []
        
        # Parse document structure
        sections = self._parse_document_structure(content)
        
        # Group sections into chunks
        chunk_groups = self._group_sections_into_chunks(sections)
        
        # Create chunks with hierarchical metadata
        for i, group in enumerate(chunk_groups):
            chunk_content = self._combine_sections(group)
            
            # Extract hierarchical information
            hierarchy_info = self._extract_hierarchy_info(group)
            
            chunk_metadata = {
                "chunk_id": i,
                "strategy": "hierarchical",
                "document_type": metadata.document_type.value,
                "total_chars": len(chunk_content),
                "hierarchy_level": hierarchy_info["max_level"],
                "section_count": len(group),
                "section_titles": hierarchy_info["titles"],
                "has_numbered_lists": self._has_numbered_lists(chunk_content),
                "parent_sections": hierarchy_info["parent_sections"]
            }
            
            start_pos = self._calculate_start_position(content, chunk_content)
            chunks.append(self._create_chunk(chunk_content, start_pos, chunk_metadata))
        
        return chunks
    
    def _parse_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical sections."""
        sections = []
        
        # Split by headers (markdown style)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = {
            "level": 0,
            "title": "",
            "content": "",
            "start_line": 0
        }
        
        for i, line in enumerate(lines):
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))  # Number of # characters
                title = header_match.group(2).strip()
                
                current_section = {
                    "level": level,
                    "title": title,
                    "content": line + "\n",
                    "start_line": i
                }
            else:
                # Add line to current section
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # If no headers found, treat as single section
        if not sections:
            sections = [{
                "level": 1,
                "title": "Document",
                "content": content,
                "start_line": 0
            }]
        
        return sections
    
    def _group_sections_into_chunks(self, sections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group sections into appropriately sized chunks."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_size = len(section["content"])
            
            # If adding this section would exceed max size
            if current_chunk and current_size + section_size > self.max_chunk_size:
                # Check if we can split the current section
                if section_size > self.max_chunk_size:
                    # Add current chunk if it has content
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_size = 0
                    
                    # Split large section
                    split_sections = self._split_large_section(section)
                    for split_section in split_sections:
                        if current_size + len(split_section["content"]) > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = []
                                current_size = 0
                        
                        current_chunk.append(split_section)
                        current_size += len(split_section["content"])
                else:
                    # Finalize current chunk
                    chunks.append(current_chunk)
                    current_chunk = [section]
                    current_size = section_size
            else:
                # Add section to current chunk
                current_chunk.append(section)
                current_size += section_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_large_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large section into smaller parts."""
        content = section["content"]
        
        # Try to split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        
        if len(paragraphs) <= 1:
            # If no paragraph breaks, split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            paragraphs = sentences
        
        # Group paragraphs into appropriately sized sections
        split_sections = []
        current_content = ""
        part_num = 1
        
        for paragraph in paragraphs:
            if current_content and len(current_content) + len(paragraph) > self.max_chunk_size:
                # Create a new section
                split_sections.append({
                    "level": section["level"],
                    "title": f"{section['title']} (Part {part_num})",
                    "content": current_content.strip(),
                    "start_line": section["start_line"]
                })
                current_content = paragraph
                part_num += 1
            else:
                if current_content:
                    current_content += "\n\n" + paragraph
                else:
                    current_content = paragraph
        
        # Add final part
        if current_content.strip():
            split_sections.append({
                "level": section["level"],
                "title": f"{section['title']} (Part {part_num})" if part_num > 1 else section["title"],
                "content": current_content.strip(),
                "start_line": section["start_line"]
            })
        
        return split_sections
    
    def _combine_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Combine sections into a single chunk content."""
        return "\n\n".join(section["content"].strip() for section in sections)
    
    def _extract_hierarchy_info(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract hierarchical information from sections."""
        levels = [section["level"] for section in sections]
        titles = [section["title"] for section in sections]
        
        # Find parent sections (sections with lower level numbers)
        parent_sections = []
        for section in sections:
            if section["level"] > 1:
                # Find the most recent parent section
                for other_section in reversed(sections):
                    if other_section["level"] < section["level"]:
                        parent_sections.append(other_section["title"])
                        break
        
        return {
            "max_level": max(levels) if levels else 1,
            "min_level": min(levels) if levels else 1,
            "titles": titles,
            "parent_sections": list(set(parent_sections))
        }
    
    def _has_numbered_lists(self, content: str) -> bool:
        """Check if content contains numbered lists."""
        return bool(re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
    
    def _calculate_start_position(self, full_content: str, chunk_content: str) -> int:
        """Calculate the start position of chunk in full content."""
        # Use first 50 characters to find position
        search_text = chunk_content[:50].strip()
        if search_text:
            pos = full_content.find(search_text)
            return max(0, pos)
        return 0 