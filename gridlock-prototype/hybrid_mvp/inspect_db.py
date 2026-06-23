import sqlite3
from pathlib import Path
DB='traffic_analytics.db'
if not Path(DB).exists():
    print('DB missing:', DB)
    raise SystemExit(1)

con=sqlite3.connect(DB)
cur=con.cursor()
cur.execute('SELECT id, violation_type, vehicle_id, frame_id, image_path, timestamp FROM violation_log ORDER BY id DESC LIMIT 50')
rows=cur.fetchall()
print(f'Found {len(rows)} rows (most recent first):')
for r in rows:
    print(r)

con.close()
