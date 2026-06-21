"""
Patch DB image_path entries to use images that actually exist on disk.
Each violation vehicle_id maps to an existing image file.
"""
import sqlite3
import os

DB = 'traffic_analytics.db'
VIOLATIONS_DIR = 'static/violations'

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Get all records
cur.execute('SELECT id, violation_type, vehicle_id, frame_id, image_path FROM violation_log ORDER BY id')
records = cur.fetchall()

# List actual files on disk
existing = os.listdir(VIOLATIONS_DIR)

updated = 0
for rec_id, vtype, vehicle_id, frame_id, image_path in records:
    # Check if the referenced file exists
    fname = os.path.basename(image_path)
    if os.path.exists(os.path.join(VIOLATIONS_DIR, fname)):
        print(f'[OK] ID {rec_id}: {fname} exists')
        continue
    
    # Find a substitute: same vehicle_id and violation_type prefix
    prefix = f'{vtype}_v{vehicle_id}_f{frame_id}_'
    candidates = [f for f in existing if f.startswith(prefix)]
    
    if not candidates:
        # Try just vehicle_id match
        prefix2 = f'{vtype}_v{vehicle_id}_'
        candidates = [f for f in existing if f.startswith(prefix2)]
    
    if candidates:
        replacement = '/static/violations/' + sorted(candidates)[0]  # pick earliest
        print(f'[PATCH] ID {rec_id}: {fname} -> {os.path.basename(replacement)}')
        cur.execute('UPDATE violation_log SET image_path=? WHERE id=?', (replacement, rec_id))
        updated += 1
    else:
        print(f'[MISS] ID {rec_id}: no substitute found for {fname}')

conn.commit()
conn.close()
print(f'\nPatched {updated} records.')
