import json

ipynb_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\notebook\Clustering.ipynb"
with open(ipynb_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx in range(37, len(notebook['cells'])):
    cell = notebook['cells'][idx]
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    print("".join(cell['source']))
    print()
