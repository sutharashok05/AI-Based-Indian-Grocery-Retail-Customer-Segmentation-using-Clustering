import json

ipynb_path = r"c:\AI-Based Indian Grocery Segmentation using Clustering\notebook\Clustering.ipynb"
with open(ipynb_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for idx, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "kmeans" in source.lower() or "scaler" in source.lower() or "cluster" in source.lower():
            print(f"--- Code Cell {idx} ---")
            print(source)
            print()
