import sqlite3, os

# check all db files
for f in ['hybrid_mvp/traffic_analytics.db', 'traffic_analytics.db']:
    if os.path.exists(f):
        print(f'=== {f} ===')
        conn = sqlite3.connect(f)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        print('Tables:', tables)
        for t in tables:
            tname = t[0]
            cur.execute(f'SELECT * FROM {tname} LIMIT 3')
            rows = cur.fetchall()
            cur.execute(f'PRAGMA table_info({tname})')
            cols = [c[1] for c in cur.fetchall()]
            print(f'  {tname} cols: {cols}')
            for row in rows:
                print(f'  row: {row}')
        conn.close()
