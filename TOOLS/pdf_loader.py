"""
PDF Loader for Zero Trading Expert
==================================
Extracts and chunks text from PDF documents for RAG.

Features:
- PDF text extraction
- Smart chunking with overlap
- Metadata extraction (title, author, pages)
- Category classification
- Direct import to trading memory
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# PDF Libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from CORE_TRADING.trading_memory import TradingMemory


class PDFLoader:
    """
    Loads and processes PDF documents for trading knowledge.
    """
    
    # Trading-related keywords for categorization
    CATEGORY_KEYWORDS = {
        "technical_analysis": [
            "rsi", "macd", "bollinger", "moving average", "indicator",
            "chart", "pattern", "support", "resistance", "trend"
        ],
        "fundamental_analysis": [
            "earnings", "revenue", "p/e ratio", "balance sheet",
            "income statement", "cash flow", "valuation"
        ],
        "risk_management": [
            "stop loss", "position size", "risk", "drawdown",
            "portfolio", "diversification", "hedge"
        ],
        "trading_strategy": [
            "strategy", "entry", "exit", "signal", "backtest",
            "momentum", "mean reversion", "breakout"
        ],
        "market_psychology": [
            "psychology", "emotion", "discipline", "fear", "greed",
            "sentiment", "behavior"
        ]
    }
    
    def __init__(self, memory: TradingMemory = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize PDF Loader.
        
        Args:
            memory: TradingMemory instance
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks for context
        """
        self.memory = memory or TradingMemory()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check available libraries
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            print("[PDF_LOADER] WARNING: No PDF library available!")
            print("[PDF_LOADER] Install: pip install PyPDF2 pdfplumber")
        elif PDFPLUMBER_AVAILABLE:
            print("[PDF_LOADER] Using pdfplumber")
        else:
            print("[PDF_LOADER] Using PyPDF2")
    
    def load_pdf(self, pdf_path: str, category: str = None) -> Dict[str, Any]:
        """
        Load and process a single PDF.
        
        Args:
            pdf_path: Path to PDF file
            category: Category override (auto-detected if not provided)
            
        Returns:
            Dict with extraction results
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return {"success": False, "error": f"File not found: {pdf_path}"}
        
        print(f"[PDF_LOADER] Loading {pdf_path.name}...")
        
        # Extract text
        if PDFPLUMBER_AVAILABLE:
            text, metadata = self._extract_with_pdfplumber(pdf_path)
        elif PYPDF2_AVAILABLE:
            text, metadata = self._extract_with_pypdf2(pdf_path)
        else:
            return {"success": False, "error": "No PDF library available"}
        
        if not text:
            return {"success": False, "error": "No text extracted from PDF"}
        
        # Clean text
        text = self._clean_text(text)
        
        # Auto-detect category
        if category is None:
            category = self._detect_category(text)
        
        # Chunk text
        chunks = self._chunk_text(text)
        
        print(f"[PDF_LOADER] Extracted {len(text)} chars, {len(chunks)} chunks")
        
        # Store chunks in memory
        stored = 0
        for i, chunk in enumerate(chunks):
            topic = f"{pdf_path.stem} - Part {i+1}"
            self.memory.store_knowledge(
                topic=topic,
                content=chunk,
                category=category
            )
            stored += 1
        
        return {
            "success": True,
            "file": pdf_path.name,
            "total_chars": len(text),
            "chunks": len(chunks),
            "stored": stored,
            "category": category,
            "metadata": metadata
        }
    
    def load_directory(self, dir_path: str, 
                      category: str = None,
                      recursive: bool = False) -> Dict[str, Any]:
        """
        Load all PDFs from a directory.
        
        Args:
            dir_path: Path to directory
            category: Category for all PDFs (auto-detect if None)
            recursive: Search subdirectories
            
        Returns:
            Dict with results for all files
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            return {"success": False, "error": f"Directory not found: {dir_path}"}
        
        # Find PDF files
        if recursive:
            pdf_files = list(dir_path.rglob("*.pdf"))
        else:
            pdf_files = list(dir_path.glob("*.pdf"))
        
        print(f"[PDF_LOADER] Found {len(pdf_files)} PDFs in {dir_path}")
        
        results = {
            "success": True,
            "total_files": len(pdf_files),
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "files": []
        }
        
        for pdf_file in pdf_files:
            result = self.load_pdf(str(pdf_file), category)
            
            if result.get("success"):
                results["processed"] += 1
                results["total_chunks"] += result.get("chunks", 0)
            else:
                results["failed"] += 1
            
            results["files"].append({
                "file": pdf_file.name,
                "success": result.get("success"),
                "chunks": result.get("chunks", 0),
                "error": result.get("error")
            })
        
        return results
    
    def load_markdown(self, md_path: str, category: str = None) -> Dict[str, Any]:
        """
        Load and process a Markdown file (for Pro-Gemini docs).
        
        Args:
            md_path: Path to Markdown file
            category: Category override
            
        Returns:
            Dict with extraction results
        """
        md_path = Path(md_path)
        
        if not md_path.exists():
            return {"success": False, "error": f"File not found: {md_path}"}
        
        print(f"[PDF_LOADER] Loading Markdown {md_path.name}...")
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return {"success": False, "error": f"Failed to read: {e}"}
        
        # Clean text
        text = self._clean_markdown(text)
        
        # Auto-detect category
        if category is None:
            category = self._detect_category(text)
        
        # Chunk text
        chunks = self._chunk_text(text)
        
        print(f"[PDF_LOADER] Extracted {len(text)} chars, {len(chunks)} chunks")
        
        # Store chunks
        stored = 0
        for i, chunk in enumerate(chunks):
            topic = f"{md_path.stem} - Part {i+1}"
            self.memory.store_knowledge(
                topic=topic,
                content=chunk,
                category=category
            )
            stored += 1
        
        return {
            "success": True,
            "file": md_path.name,
            "total_chars": len(text),
            "chunks": len(chunks),
            "stored": stored,
            "category": category
        }
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> tuple:
        """Extract text using pdfplumber."""
        text = ""
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = {
                    "pages": len(pdf.pages),
                    "metadata": pdf.metadata or {}
                }
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"[PDF_LOADER] pdfplumber error: {e}")
            return "", {}
        
        return text, metadata
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> tuple:
        """Extract text using PyPDF2."""
        text = ""
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                metadata = {
                    "pages": len(reader.pages),
                    "metadata": dict(reader.metadata) if reader.metadata else {}
                }
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"[PDF_LOADER] PyPDF2 error: {e}")
            return "", {}
        
        return text, metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers patterns
        text = re.sub(r'\b\d+\s*\|\s*Page\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bPage\s*\d+\s*(of\s*\d+)?\b', '', text, flags=re.IGNORECASE)
        
        # Remove headers/footers (often repeated)
        # This is a simple heuristic
        lines = text.split('. ')
        if len(lines) > 10:
            # Remove very short repeated lines
            line_counts = {}
            for line in lines:
                line_stripped = line.strip()[:50]
                line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
            
            # Filter out lines that appear too often (likely headers/footers)
            filtered_lines = []
            for line in lines:
                line_stripped = line.strip()[:50]
                if line_counts.get(line_stripped, 0) < 5:
                    filtered_lines.append(line)
            
            text = '. '.join(filtered_lines)
        
        return text.strip()
    
    def _clean_markdown(self, text: str) -> str:
        """Clean Markdown text."""
        # Remove code blocks (keep prose)
        text = re.sub(r'```[\s\S]*?```', '[code block removed]', text)
        
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove images
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean headers (keep text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Clean emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def _detect_category(self, text: str) -> str:
        """Auto-detect category based on content."""
        text_lower = text.lower()
        
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = score
        
        if not scores or max(scores.values()) == 0:
            return "general"
        
        return max(scores, key=scores.get)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                if chunks and self.chunk_overlap > 0:
                    overlap_text = chunks[-1][-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def export_to_jsonl(self, output_path: str = None) -> str:
        """
        Export loaded knowledge to JSONL format.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to created file
        """
        output_path = output_path or str(
            Path(__file__).parent.parent / "DATASETS" / "pdf_extracts.jsonl"
        )
        
        # Get all knowledge from memory
        if not self.memory.technical_knowledge:
            print("[PDF_LOADER] No knowledge to export")
            return ""
        
        count = self.memory.technical_knowledge.count()
        if count == 0:
            print("[PDF_LOADER] No knowledge to export")
            return ""
        
        results = self.memory.technical_knowledge.get(limit=count)
        
        if not results or not results.get('documents'):
            return ""
        
        # Write to JSONL
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(results['documents']):
                entry = {
                    "document": doc,
                    "metadata": results['metadatas'][i] if results.get('metadatas') else {},
                    "id": results['ids'][i] if results.get('ids') else f"doc_{i}"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"[PDF_LOADER] Exported {len(results['documents'])} chunks to {output_path}")
        
        return str(output_path)


# ===== CLI Interface =====

def main():
    """Command-line interface for PDF loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load PDFs into ZTE knowledge base")
    parser.add_argument('path', type=str, nargs='?',
                       help='PDF file or directory path')
    parser.add_argument('--category', type=str, default=None,
                       help='Category for the content')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories')
    parser.add_argument('--markdown', action='store_true',
                       help='Load as Markdown instead of PDF')
    parser.add_argument('--export', action='store_true',
                       help='Export to JSONL after loading')
    parser.add_argument('--load-pro-gemini-docs', action='store_true',
                       help='Load Pro-Gemini-Trade documentation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PDF Loader - Zero Trading Expert")
    print("="*60)
    
    loader = PDFLoader()
    
    # Load Pro-Gemini docs if requested
    if args.load_pro_gemini_docs:
        docs_path = Path("C:/Vs-Pro/pro-gemini-traed/docs")
        if docs_path.exists():
            print(f"\n--- Loading Pro-Gemini-Trade Documentation ---")
            md_files = list(docs_path.glob("*.md"))
            for md_file in md_files:
                result = loader.load_markdown(str(md_file))
                status = "✓" if result.get("success") else "✗"
                print(f"  {status} {md_file.name}: {result.get('chunks', 0)} chunks")
        else:
            print(f"[ERROR] Docs directory not found: {docs_path}")
    
    # Load specified path
    elif args.path:
        path = Path(args.path)
        
        if path.is_file():
            if args.markdown or path.suffix.lower() == '.md':
                result = loader.load_markdown(str(path), args.category)
            else:
                result = loader.load_pdf(str(path), args.category)
            
            print("\n--- Load Result ---")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        elif path.is_dir():
            result = loader.load_directory(str(path), args.category, args.recursive)
            
            print("\n--- Directory Load Result ---")
            print(f"  Total files: {result['total_files']}")
            print(f"  Processed: {result['processed']}")
            print(f"  Failed: {result['failed']}")
            print(f"  Total chunks: {result['total_chunks']}")
        
        else:
            print(f"[ERROR] Path not found: {path}")
    
    else:
        print("\nUsage: python pdf_loader.py <path> [--category CATEGORY]")
        print("       python pdf_loader.py --load-pro-gemini-docs")
        return
    
    # Export if requested
    if args.export:
        print("\n--- Exporting to JSONL ---")
        output = loader.export_to_jsonl()
        print(f"  Exported to: {output}")
    
    # Show memory stats
    print("\n--- Memory Statistics ---")
    stats = loader.memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

