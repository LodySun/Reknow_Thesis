import sqlite3
import pandas as pd

db_path = "db_path" 
conn = sqlite3.connect(db_path)

tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
print("Tables:")
print(tables)

for table_name in tables["name"].tolist():
    print(f"\n=== Table: {table_name} ===")

    n_rows = pd.read_sql_query(f'SELECT COUNT(*) AS n_rows FROM "{table_name}";', conn)
    print("Row count:")
    print(n_rows.to_string(index=False))

    schema = pd.read_sql_query(f'PRAGMA table_info("{table_name}");', conn)
    print("Schema:")
    print(schema[["cid", "name", "type", "notnull", "pk"]].to_string(index=False))

    preview = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 5;', conn)
    print("Preview (first 5 rows):")
    if preview.empty:
        print("(empty table)")
    else:
        print(preview.to_string(index=False))

conn.close()
