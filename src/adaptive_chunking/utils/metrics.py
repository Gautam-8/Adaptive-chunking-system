"""
Metrics utilities for analyzing chunk quality and performance.
"""

import statistics
from typing import List, Dict, Any, Optional
from collections import Counter


def calculate_chunk_metrics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a list of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary containing various metrics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "median_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "std_chunk_size": 0,
            "strategy_distribution": {},
            "document_type_distribution": {},
            "total_characters": 0
        }
    
    # Extract chunk sizes
    chunk_sizes = []
    strategies = []
    document_types = []
    total_chars = 0
    
    for chunk in chunks:
        if 'content' in chunk:
            size = len(chunk['content'])
            chunk_sizes.append(size)
            total_chars += size
        elif 'total_chars' in chunk.get('metadata', {}):
            size = chunk['metadata']['total_chars']
            chunk_sizes.append(size)
            total_chars += size
        
        # Extract strategy and document type from metadata
        metadata = chunk.get('metadata', {})
        if 'strategy' in metadata:
            strategies.append(metadata['strategy'])
        if 'document_type' in metadata:
            document_types.append(metadata['document_type'])
    
    # Calculate basic statistics
    metrics = {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": statistics.mean(chunk_sizes) if chunk_sizes else 0,
        "median_chunk_size": statistics.median(chunk_sizes) if chunk_sizes else 0,
        "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "std_chunk_size": statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
        "strategy_distribution": dict(Counter(strategies)),
        "document_type_distribution": dict(Counter(document_types))
    }
    
    # Calculate additional metrics
    if chunk_sizes:
        metrics.update({
            "size_percentiles": {
                "25th": _percentile(chunk_sizes, 25),
                "50th": _percentile(chunk_sizes, 50),
                "75th": _percentile(chunk_sizes, 75),
                "95th": _percentile(chunk_sizes, 95)
            },
            "size_distribution": _size_distribution(chunk_sizes),
            "efficiency_score": _calculate_efficiency_score(chunk_sizes)
        })
    
    return metrics


def calculate_document_metrics(processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for document processing results.
    
    Args:
        processing_results: List of processing result dictionaries
        
    Returns:
        Dictionary containing document-level metrics
    """
    if not processing_results:
        return {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "success_rate": 0.0,
            "avg_processing_time": 0.0,
            "avg_chunks_per_document": 0.0
        }
    
    successful = [r for r in processing_results if r.get('success', False)]
    failed = [r for r in processing_results if not r.get('success', False)]
    
    processing_times = [r.get('processing_time', 0) for r in processing_results]
    chunks_per_doc = [len(r.get('chunks', [])) for r in successful]
    
    return {
        "total_documents": len(processing_results),
        "successful_documents": len(successful),
        "failed_documents": len(failed),
        "success_rate": len(successful) / len(processing_results) * 100,
        "avg_processing_time": statistics.mean(processing_times) if processing_times else 0,
        "median_processing_time": statistics.median(processing_times) if processing_times else 0,
        "avg_chunks_per_document": statistics.mean(chunks_per_doc) if chunks_per_doc else 0,
        "median_chunks_per_document": statistics.median(chunks_per_doc) if chunks_per_doc else 0
    }


def analyze_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality of chunks based on various criteria.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary containing quality metrics
    """
    if not chunks:
        return {"quality_score": 0.0, "issues": []}
    
    issues = []
    quality_factors = []
    
    # Check for empty chunks
    empty_chunks = sum(1 for c in chunks if not c.get('content', '').strip())
    if empty_chunks > 0:
        issues.append(f"{empty_chunks} empty chunks found")
        quality_factors.append(0.0)
    else:
        quality_factors.append(1.0)
    
    # Check for very small chunks (might indicate over-chunking)
    small_chunks = sum(1 for c in chunks if len(c.get('content', '')) < 50)
    small_ratio = small_chunks / len(chunks)
    if small_ratio > 0.2:  # More than 20% are very small
        issues.append(f"{small_ratio:.1%} of chunks are very small (<50 chars)")
        quality_factors.append(max(0.0, 1.0 - small_ratio))
    else:
        quality_factors.append(1.0)
    
    # Check for very large chunks (might indicate under-chunking)
    large_chunks = sum(1 for c in chunks if len(c.get('content', '')) > 2000)
    large_ratio = large_chunks / len(chunks)
    if large_ratio > 0.1:  # More than 10% are very large
        issues.append(f"{large_ratio:.1%} of chunks are very large (>2000 chars)")
        quality_factors.append(max(0.0, 1.0 - large_ratio))
    else:
        quality_factors.append(1.0)
    
    # Check for consistent chunk sizes
    sizes = [len(c.get('content', '')) for c in chunks]
    if len(sizes) > 1:
        cv = statistics.stdev(sizes) / statistics.mean(sizes)  # Coefficient of variation
        if cv > 1.0:  # High variation
            issues.append(f"High variation in chunk sizes (CV: {cv:.2f})")
            quality_factors.append(max(0.0, 1.0 - (cv - 1.0)))
        else:
            quality_factors.append(1.0)
    else:
        quality_factors.append(1.0)
    
    # Overall quality score
    quality_score = statistics.mean(quality_factors) if quality_factors else 0.0
    
    return {
        "quality_score": quality_score,
        "issues": issues,
        "empty_chunks": empty_chunks,
        "small_chunks": small_chunks,
        "large_chunks": large_chunks,
        "size_variation": cv if len(sizes) > 1 else 0.0
    }


def compare_chunking_strategies(results_by_strategy: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compare different chunking strategies based on their results.
    
    Args:
        results_by_strategy: Dictionary mapping strategy names to their results
        
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {}
    
    for strategy_name, results in results_by_strategy.items():
        if not results:
            continue
            
        # Extract chunks from results
        all_chunks = []
        for result in results:
            all_chunks.extend(result.get('chunks', []))
        
        # Calculate metrics for this strategy
        chunk_metrics = calculate_chunk_metrics(all_chunks)
        quality_metrics = analyze_chunk_quality(all_chunks)
        
        comparison[strategy_name] = {
            "total_chunks": chunk_metrics["total_chunks"],
            "avg_chunk_size": chunk_metrics["avg_chunk_size"],
            "quality_score": quality_metrics["quality_score"],
            "efficiency_score": chunk_metrics.get("efficiency_score", 0.0),
            "issues": quality_metrics["issues"]
        }
    
    # Find the best strategy
    best_strategy = None
    best_score = 0.0
    
    for strategy, metrics in comparison.items():
        # Combined score: quality * efficiency
        combined_score = metrics["quality_score"] * metrics["efficiency_score"]
        if combined_score > best_score:
            best_score = combined_score
            best_strategy = strategy
    
    return {
        "strategies": comparison,
        "best_strategy": best_strategy,
        "best_score": best_score
    }


def _percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of a list of numbers."""
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * percentile / 100
    f = int(k)
    c = k - f
    
    if f == len(sorted_data) - 1:
        return sorted_data[f]
    else:
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c


def _size_distribution(sizes: List[int]) -> Dict[str, int]:
    """Categorize chunk sizes into distribution buckets."""
    distribution = {
        "very_small": 0,    # < 100 chars
        "small": 0,         # 100-500 chars
        "medium": 0,        # 500-1000 chars
        "large": 0,         # 1000-2000 chars
        "very_large": 0     # > 2000 chars
    }
    
    for size in sizes:
        if size < 100:
            distribution["very_small"] += 1
        elif size < 500:
            distribution["small"] += 1
        elif size < 1000:
            distribution["medium"] += 1
        elif size < 2000:
            distribution["large"] += 1
        else:
            distribution["very_large"] += 1
    
    return distribution


def _calculate_efficiency_score(sizes: List[int], target_size: int = 1000) -> float:
    """
    Calculate efficiency score based on how close chunks are to target size.
    
    Args:
        sizes: List of chunk sizes
        target_size: Target chunk size
        
    Returns:
        Efficiency score between 0 and 1
    """
    if not sizes:
        return 0.0
    
    # Calculate deviation from target size
    deviations = [abs(size - target_size) / target_size for size in sizes]
    avg_deviation = statistics.mean(deviations)
    
    # Convert to efficiency score (lower deviation = higher efficiency)
    efficiency = max(0.0, 1.0 - avg_deviation)
    return efficiency 