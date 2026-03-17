import json

with open('face_mask_detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extract.out', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            out.write(''.join(cell['source']))
            out.write('\n# --- \n')
