import json

ipynb_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\notebook\Clustering.ipynb"
with open(ipynb_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        print(f"--- Markdown Cell {idx} ---")
        print("".join(cell['source']))
        print()
