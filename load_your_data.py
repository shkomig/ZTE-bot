"""Load user's data files into ZTE memory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

# Import modules
def import_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pdf_mod = import_mod("pdf_loader", "TOOLS/pdf_loader.py")
PDFLoader = pdf_mod.PDFLoader

print("="*60)
print("LOADING YOUR DATA INTO ZTE")
print("="*60)

loader = PDFLoader()

# Load Markdown files
md_path = Path("YOUR_DATA/Documents")
if md_path.exists():
    print("\n--- Loading Markdown Documents ---")
    for md_file in md_path.glob("*.md"):
        print(f"  Loading: {md_file.name}")
        result = loader.load_markdown(str(md_file), category="llm_trading_research")
        chunks = result.get("chunks", 0)
        print(f"    -> {chunks} chunks loaded")

# Load PDFs
pdf_path = Path("YOUR_DATA/PDFs")
if pdf_path.exists():
    print("\n--- Loading PDF Documents ---")
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        result = loader.load_pdf(str(pdf_file), category="trading_research")
        if result.get("success"):
            print(f"    -> {result.get('chunks', 0)} chunks loaded")
        else:
            print(f"    -> Error: {result.get('error', 'Unknown')}")

# Final stats
print("\n" + "="*60)
print("FINAL MEMORY STATISTICS")
print("="*60)
stats = loader.memory.get_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

total = sum(stats.values())
print(f"\n  TOTAL ITEMS: {total}")
print("="*60)

