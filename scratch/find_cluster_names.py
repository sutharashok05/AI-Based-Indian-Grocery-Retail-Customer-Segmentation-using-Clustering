import json

ipynb_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\notebook\Clustering.ipynb"
with open(ipynb_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook['cells']):
    source_str = "".join(cell['source'])
    if "Budget & Discount Products" in source_str or "Premium & Popular Products" in source_str:
        print(f"--- Cell {idx} ({cell['cell_type']}) ---")
        print(source_str)
        print()
