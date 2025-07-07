"""
Document classifier for identifying document types and structure patterns.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class DocumentType(Enum):
    """Enumeration of supported document types."""
    
    TECHNICAL_DOC = "technical_doc"
    API_REFERENCE = "api_reference"
    SUPPORT_TICKET = "support_ticket"
    POLICY_DOCUMENT = "policy_document"
    TUTORIAL = "tutorial"
    CODE_DOCUMENTATION = "code_documentation"
    TROUBLESHOOTING = "troubleshooting"
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Metadata extracted from document analysis."""
    
    document_type: DocumentType
    confidence: float
    structure_patterns: List[str]
    language: Optional[str] = None
    has_code_blocks: bool = False
    has_numbered_steps: bool = False
    has_hierarchical_structure: bool = False


class DocumentClassifier:
    """
    Classifies documents based on content patterns and structure.
    
    This is a simple rule-based classifier that identifies document types
    based on content patterns, keywords, and structural elements.
    """
    
    def __init__(self):
        """Initialize the document classifier with pattern rules."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[DocumentType, List[str]]:
        """Initialize classification patterns for different document types."""
        return {
            DocumentType.API_REFERENCE: [
                r'\b(GET|POST|PUT|DELETE|PATCH)\b',
                r'\b(endpoint|API|REST|GraphQL)\b',
                r'\b(request|response|parameter)\b',
                r'```\s*(json|curl|http)',
                r'\b(authentication|authorization)\b'
            ],
            DocumentType.TECHNICAL_DOC: [
                r'\b(architecture|design|specification)\b',
                r'\b(requirements|implementation)\b',
                r'\b(system|component|module)\b',
                r'```\s*(yaml|xml|json)',
                r'\b(configuration|setup|installation)\b'
            ],
            DocumentType.SUPPORT_TICKET: [
                r'\b(issue|problem|error|bug)\b',
                r'\b(ticket|incident|case)\b',
                r'\b(reported|customer|user)\b',
                r'\b(priority|severity|status)\b',
                r'\b(resolution|workaround|fix)\b'
            ],
            DocumentType.POLICY_DOCUMENT: [
                r'\b(policy|procedure|guideline)\b',
                r'\b(compliance|regulation|standard)\b',
                r'\b(must|shall|should|required)\b',
                r'\b(approval|authorization|permission)\b',
                r'\b(violation|penalty|consequence)\b'
            ],
            DocumentType.TUTORIAL: [
                r'\b(tutorial|guide|walkthrough)\b',
                r'\b(step|lesson|chapter)\b',
                r'\b(learn|how to|getting started)\b',
                r'^\s*\d+\.\s+',  # Numbered steps
                r'\b(example|practice|exercise)\b'
            ],
            DocumentType.CODE_DOCUMENTATION: [
                r'```\s*(python|java|javascript|typescript|c\+\+|c#)',
                r'\b(function|method|class|variable)\b',
                r'\b(import|include|require)\b',
                r'\b(parameter|return|throws)\b',
                r'@\w+\s*\(',  # Decorators/annotations
            ],
            DocumentType.TROUBLESHOOTING: [
                r'\b(troubleshoot|debug|diagnose)\b',
                r'\b(symptom|cause|solution)\b',
                r'\b(error|warning|failure)\b',
                r'\b(check|verify|ensure)\b',
                r'^\s*\d+\.\s+.*\b(step|check|verify)\b'
            ]
        }
    
    def classify(self, content: str, metadata: Optional[Dict] = None) -> DocumentMetadata:
        """
        Classify a document based on its content and optional metadata.
        
        Args:
            content: The document content as a string
            metadata: Optional metadata dictionary (filename, source, etc.)
            
        Returns:
            DocumentMetadata with classification results
        """
        scores = {}
        
        # Score each document type based on pattern matches
        for doc_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                score += matches
            scores[doc_type] = score
        
        # Find the best match
        if not scores or max(scores.values()) == 0:
            best_type = DocumentType.UNKNOWN
            confidence = 0.0
        else:
            best_type = max(scores.keys(), key=lambda k: scores[k])
            total_matches = sum(scores.values())
            confidence = scores[best_type] / total_matches if total_matches > 0 else 0.0
        
        # Analyze document structure
        structure_patterns = self._analyze_structure(content)
        
        return DocumentMetadata(
            document_type=best_type,
            confidence=confidence,
            structure_patterns=structure_patterns,
            language=self._detect_language(content),
            has_code_blocks=self._has_code_blocks(content),
            has_numbered_steps=self._has_numbered_steps(content),
            has_hierarchical_structure=self._has_hierarchical_structure(content)
        )
    
    def _analyze_structure(self, content: str) -> List[str]:
        """Analyze the structural patterns in the document."""
        patterns = []
        
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
            patterns.append("markdown_headers")
        
        if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
            patterns.append("numbered_list")
        
        if re.search(r'^\s*[-*+]\s+', content, re.MULTILINE):
            patterns.append("bullet_list")
        
        if re.search(r'```[\s\S]*?```', content):
            patterns.append("code_blocks")
        
        if re.search(r'\|.*\|', content):
            patterns.append("tables")
        
        return patterns
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Simple language detection for code blocks."""
        code_block_pattern = r'```(\w+)'
        matches = re.findall(code_block_pattern, content)
        if matches:
            return matches[0]  # Return the first detected language
        return None
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if document contains code blocks."""
        return bool(re.search(r'```[\s\S]*?```', content))
    
    def _has_numbered_steps(self, content: str) -> bool:
        """Check if document contains numbered steps."""
        return bool(re.search(r'^\s*\d+\.\s+', content, re.MULTILINE))
    
    def _has_hierarchical_structure(self, content: str) -> bool:
        """Check if document has hierarchical structure (headers)."""
        return bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE)) 