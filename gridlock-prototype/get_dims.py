import json, urllib.request, cv2, os
data = json.loads(urllib.request.urlopen('http://127.0.0.1:8001/api/violations?limit=10').read())
base_dir = '.'
print('ID | TYPE | DIMENSIONS | FILE')
for v in data['violations']:
    path = os.path.join(base_dir, v['image_url'].lstrip('/'))
    if os.path.exists(path):
        img = cv2.imread(path)
        h, w = img.shape[:2] if img is not None else (0,0)
        print(f"{v['id']:2d} | {v['violation_type']:13s} | {w:3d}x{h:3d} | {path}")
    else:
        print(f"{v['id']:2d} | {v['violation_type']:13s} | NOT FOUND | {path}")
