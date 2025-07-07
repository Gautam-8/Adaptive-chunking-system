"""
File utilities for the adaptive chunking system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_document(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a document from various file formats.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary containing document content and metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and load accordingly
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.json':
            return _load_json(file_path)
        elif suffix in ['.yaml', '.yml']:
            return _load_yaml(file_path)
        elif suffix == '.txt':
            return _load_text(file_path)
        elif suffix == '.md':
            return _load_markdown(file_path)
        else:
            # Default to text loading
            return _load_text(file_path)
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise


def _load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If it's a structured document, extract content
    if isinstance(data, dict):
        content = data.get('content', str(data))
        metadata = {k: v for k, v in data.items() if k != 'content'}
    else:
        content = str(data)
        metadata = {}
    
    return {
        'content': content,
        'metadata': {
            'filename': file_path.name,
            'file_type': 'json',
            'file_size': file_path.stat().st_size,
            **metadata
        }
    }


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # If it's a structured document, extract content
    if isinstance(data, dict):
        content = data.get('content', str(data))
        metadata = {k: v for k, v in data.items() if k != 'content'}
    else:
        content = str(data)
        metadata = {}
    
    return {
        'content': content,
        'metadata': {
            'filename': file_path.name,
            'file_type': 'yaml',
            'file_size': file_path.stat().st_size,
            **metadata
        }
    }


def _load_text(file_path: Path) -> Dict[str, Any]:
    """Load plain text document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    return {
        'content': content,
        'metadata': {
            'filename': file_path.name,
            'file_type': 'text',
            'file_size': file_path.stat().st_size
        }
    }


def _load_markdown(file_path: Path) -> Dict[str, Any]:
    """Load Markdown document."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    return {
        'content': content,
        'metadata': {
            'filename': file_path.name,
            'file_type': 'markdown',
            'file_size': file_path.stat().st_size
        }
    }


def save_chunks(chunks: List[Dict[str, Any]], 
                output_path: Union[str, Path],
                format: str = 'json') -> None:
    """
    Save chunks to a file.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the chunks
        format: Output format ('json', 'yaml', 'jsonl')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    elif format == 'yaml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(chunks, f, default_flow_style=False, allow_unicode=True)
    
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load chunks from a file.
    
    Args:
        file_path: Path to the chunks file
        
    Returns:
        List of chunk dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif suffix in ['.yaml', '.yml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    elif suffix == '.jsonl':
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def create_sample_documents(output_dir: Union[str, Path]) -> None:
    """
    Create sample documents for testing the adaptive chunking system.
    
    Args:
        output_dir: Directory to create sample documents
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample API documentation
    api_doc = """# User Management API

## Overview
This API provides endpoints for managing user accounts in the system.

## Authentication
All API requests require authentication using Bearer tokens.

```http
Authorization: Bearer <your-token>
```

## Endpoints

### GET /api/users
Retrieve a list of all users.

**Parameters:**
- `limit` (optional): Maximum number of users to return (default: 50)
- `offset` (optional): Number of users to skip (default: 0)

**Response:**
```json
{
  "users": [
    {
      "id": 1,
      "username": "john_doe",
      "email": "john@example.com",
      "created_at": "2023-01-01T00:00:00Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### POST /api/users
Create a new user account.

**Request Body:**
```json
{
  "username": "new_user",
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "id": 2,
  "username": "new_user",
  "email": "user@example.com",
  "created_at": "2023-01-02T00:00:00Z"
}
```
"""
    
    # Sample troubleshooting guide
    troubleshooting_doc = """# Database Connection Issues

## Problem Description
Users are experiencing intermittent database connection failures, resulting in application errors and timeouts.

## Symptoms
- Application throws "Connection timeout" errors
- Database queries fail randomly
- Users see "Service unavailable" messages
- Connection pool exhaustion warnings in logs

## Troubleshooting Steps

1. **Check Database Server Status**
   - Verify the database server is running
   - Check system resources (CPU, memory, disk space)
   - Review database server logs for errors

2. **Verify Network Connectivity**
   - Test network connectivity between application and database
   - Check for firewall rules blocking connections
   - Verify DNS resolution for database hostname

3. **Review Connection Pool Settings**
   - Check maximum connection pool size
   - Verify connection timeout settings
   - Review connection validation queries

4. **Monitor Database Performance**
   - Check for long-running queries
   - Review query execution plans
   - Monitor database locks and deadlocks

## Common Solutions

### Increase Connection Pool Size
```python
# Example configuration
DATABASE_CONFIG = {
    'max_connections': 100,
    'connection_timeout': 30,
    'pool_recycle': 3600
}
```

### Optimize Query Performance
- Add appropriate indexes
- Rewrite inefficient queries
- Use connection pooling
- Implement query caching

## Prevention
- Regular database maintenance
- Monitor connection metrics
- Set up alerting for connection failures
- Implement proper error handling and retry logic
"""
    
    # Sample policy document
    policy_doc = """# Data Security Policy

## 1. Purpose
This policy establishes the requirements for protecting sensitive data within the organization.

## 2. Scope
This policy applies to all employees, contractors, and third-party vendors who have access to organizational data.

## 3. Data Classification

### 3.1 Public Data
Data that can be freely shared without risk to the organization.

### 3.2 Internal Data
Data intended for use within the organization but not for external distribution.

### 3.3 Confidential Data
Sensitive data that requires protection and has restricted access.

### 3.4 Restricted Data
Highly sensitive data with the most stringent access controls.

## 4. Access Controls

### 4.1 Authentication Requirements
- All users must use strong, unique passwords
- Multi-factor authentication is required for sensitive systems
- Password must be changed every 90 days

### 4.2 Authorization Principles
- Access shall be granted on a need-to-know basis
- Principle of least privilege must be followed
- Regular access reviews are mandatory

## 5. Data Handling

### 5.1 Data Storage
- Confidential data must be encrypted at rest
- Backups must be encrypted and stored securely
- Data retention policies must be followed

### 5.2 Data Transmission
- All data in transit must be encrypted
- Secure protocols (HTTPS, SFTP) must be used
- Email encryption required for sensitive data

## 6. Compliance and Monitoring

### 6.1 Monitoring Requirements
- All data access must be logged
- Regular security audits are required
- Incident response procedures must be followed

### 6.2 Violations
Violations of this policy may result in disciplinary action, up to and including termination of employment.

## 7. Policy Review
This policy shall be reviewed annually and updated as necessary.
"""
    
    # Save sample documents
    samples = [
        ('api_documentation.md', api_doc),
        ('troubleshooting_guide.md', troubleshooting_doc),
        ('data_security_policy.md', policy_doc)
    ]
    
    for filename, content in samples:
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    logger.info(f"Created {len(samples)} sample documents in {output_dir}") 