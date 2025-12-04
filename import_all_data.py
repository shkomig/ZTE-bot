"""
Import All Data - Zero Trading Expert
=====================================
Master script to import all data sources into ZTE memory.

Run this after initial setup to populate the knowledge base.

Usage:
    python import_all_data.py
    python import_all_data.py --trades-only
    python import_all_data.py --docs-only
"""

import sys
from pathlib import Path
import argparse

# Add current directory to path
ZTE_PATH = str(Path(__file__).parent)
sys.path.insert(0, ZTE_PATH)

# Import with full path handling
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules
trading_memory_mod = import_module_from_path("trading_memory", Path(ZTE_PATH) / "CORE_TRADING" / "trading_memory.py")
TradingMemory = trading_memory_mod.TradingMemory

trade_importer_mod = import_module_from_path("trade_log_importer", Path(ZTE_PATH) / "TOOLS" / "trade_log_importer.py")
TradeLogImporter = trade_importer_mod.TradeLogImporter

pdf_loader_mod = import_module_from_path("pdf_loader", Path(ZTE_PATH) / "TOOLS" / "pdf_loader.py")
PDFLoader = pdf_loader_mod.PDFLoader


def import_trades(memory: TradingMemory):
    """Import trades from Pro-Gemini-Trade CSV."""
    print("\n" + "="*60)
    print("IMPORTING TRADES FROM PRO-GEMINI-TRADE")
    print("="*60)
    
    importer = TradeLogImporter(memory)
    result = importer.import_csv()
    
    print(f"\n  Success: {result.get('success')}")
    print(f"  Imported: {result.get('imported', 0)}")
    print(f"  Skipped: {result.get('skipped', 0)}")
    print(f"  Total in file: {result.get('total_in_file', 0)}")
    
    # Export to JSONL
    if result.get('success') and result.get('imported', 0) > 0:
        output = importer.export_to_jsonl()
        print(f"  Exported to: {output}")
    
    return result


def import_pro_gemini_docs(memory: TradingMemory):
    """Import documentation from Pro-Gemini-Trade."""
    print("\n" + "="*60)
    print("IMPORTING PRO-GEMINI-TRADE DOCUMENTATION")
    print("="*60)
    
    docs_path = Path("C:/Vs-Pro/pro-gemini-traed/docs")
    
    if not docs_path.exists():
        print(f"  WARNING: Docs directory not found: {docs_path}")
        return {"success": False}
    
    loader = PDFLoader(memory)
    
    md_files = list(docs_path.glob("*.md"))
    print(f"  Found {len(md_files)} Markdown files")
    
    results = {"success": True, "processed": 0, "chunks": 0}
    
    for md_file in md_files:
        result = loader.load_markdown(str(md_file))
        if result.get("success"):
            results["processed"] += 1
            results["chunks"] += result.get("chunks", 0)
            print(f"    ✓ {md_file.name}: {result.get('chunks', 0)} chunks")
        else:
            print(f"    ✗ {md_file.name}: {result.get('error', 'Unknown error')}")
    
    print(f"\n  Total processed: {results['processed']}/{len(md_files)}")
    print(f"  Total chunks: {results['chunks']}")
    
    return results


def load_base_knowledge(memory: TradingMemory):
    """Load base trading knowledge from dataset."""
    print("\n" + "="*60)
    print("LOADING BASE TRADING KNOWLEDGE")
    print("="*60)
    
    import json
    
    dataset_path = Path(__file__).parent / "DATASETS" / "trading_knowledge.jsonl"
    
    if not dataset_path.exists():
        print(f"  WARNING: Dataset not found: {dataset_path}")
        return {"success": False}
    
    count = 0
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                memory.store_knowledge(
                    topic=item.get('input', 'Unknown')[:100],
                    content=item.get('output', ''),
                    category=item.get('instruction', 'general')[:50]
                )
                count += 1
            except Exception as e:
                print(f"  Error loading item: {e}")
    
    print(f"  Loaded {count} knowledge items")
    
    return {"success": True, "count": count}


def load_patterns(memory: TradingMemory):
    """Load common trading patterns into memory."""
    print("\n" + "="*60)
    print("LOADING TRADING PATTERNS")
    print("="*60)
    
    patterns = [
        {
            "name": "Double Bottom",
            "description": "Bullish reversal pattern forming 'W' shape. Entry on breakout above neckline. Stop below the bottoms. Target equals pattern height projected above breakout.",
            "success_rate": 0.65
        },
        {
            "name": "Double Top",
            "description": "Bearish reversal pattern forming 'M' shape. Entry on breakdown below neckline. Stop above the tops. Target equals pattern height projected below breakdown.",
            "success_rate": 0.62
        },
        {
            "name": "Head and Shoulders",
            "description": "Bearish reversal with left shoulder, head (higher), right shoulder. Entry on neckline break. Stop above right shoulder. Target equals head-to-neckline distance.",
            "success_rate": 0.70
        },
        {
            "name": "Inverse Head and Shoulders",
            "description": "Bullish reversal with left shoulder, head (lower), right shoulder. Entry on neckline breakout. Stop below right shoulder. Target equals head-to-neckline distance.",
            "success_rate": 0.70
        },
        {
            "name": "Bull Flag",
            "description": "Continuation pattern: sharp rally (pole) followed by tight consolidation (flag). Entry on breakout above flag. Stop below flag. Target equals pole length.",
            "success_rate": 0.68
        },
        {
            "name": "Bear Flag",
            "description": "Continuation pattern: sharp decline (pole) followed by tight consolidation (flag). Entry on breakdown below flag. Stop above flag. Target equals pole length.",
            "success_rate": 0.65
        },
        {
            "name": "Ascending Triangle",
            "description": "Typically bullish: flat resistance with rising support. Shows increasing buyer pressure. Entry on resistance breakout. Stop below rising support.",
            "success_rate": 0.72
        },
        {
            "name": "Descending Triangle",
            "description": "Typically bearish: flat support with declining resistance. Shows increasing seller pressure. Entry on support breakdown. Stop above falling resistance.",
            "success_rate": 0.70
        },
        {
            "name": "Cup and Handle",
            "description": "Bullish continuation: U-shaped cup recovery followed by small handle pullback. Entry on handle breakout. Stop below handle. Target equals cup depth.",
            "success_rate": 0.65
        },
        {
            "name": "Gap and Go",
            "description": "Momentum pattern: stock gaps up >3% with high volume and continues upward. Entry above pre-market high. Stop below gap low. Ride momentum.",
            "success_rate": 0.60
        }
    ]
    
    count = 0
    for pattern in patterns:
        memory.store_pattern(
            pattern_name=pattern["name"],
            description=pattern["description"],
            success_rate=pattern["success_rate"]
        )
        count += 1
        print(f"    + {pattern['name']}")
    
    print(f"\n  Loaded {count} patterns")
    
    return {"success": True, "count": count}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Import all data into ZTE")
    parser.add_argument('--trades-only', action='store_true',
                       help='Import only trades')
    parser.add_argument('--docs-only', action='store_true',
                       help='Import only documentation')
    parser.add_argument('--knowledge-only', action='store_true',
                       help='Import only base knowledge')
    parser.add_argument('--patterns-only', action='store_true',
                       help='Import only patterns')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ZERO TRADING EXPERT - DATA IMPORT")
    print("="*60)
    
    # Initialize memory
    memory = TradingMemory()
    
    # Import based on flags
    if args.trades_only:
        import_trades(memory)
    elif args.docs_only:
        import_pro_gemini_docs(memory)
    elif args.knowledge_only:
        load_base_knowledge(memory)
    elif args.patterns_only:
        load_patterns(memory)
    else:
        # Import everything
        load_base_knowledge(memory)
        load_patterns(memory)
        import_trades(memory)
        import_pro_gemini_docs(memory)
    
    # Final stats
    print("\n" + "="*60)
    print("FINAL MEMORY STATISTICS")
    print("="*60)
    
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    win_rate = memory.get_win_rate()
    print(f"\n  Win Rate: {win_rate:.1%}")
    
    print("\n" + "="*60)
    print("DATA IMPORT COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Create the model: ollama create zero-trading-expert -f MODELS/Modelfile.trading-expert")
    print("  2. Start the server: python api_server_trading.py")
    print("  3. Test: curl http://localhost:5001/api/health")


if __name__ == "__main__":
    main()

