import os
import pandas as pd
import pymysql

# Database connection settings
conn = pymysql.connect(
    host='relational.fel.cvut.cz',
    port=3306,
    user='guest',
    password='ctu-relational',
    database='Basketball_men'
)

# Create output directories
csv_dir = 'datasets/basketball_csvs'
sql_dir = 'datasets/basketball_csvs/schemas'
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(sql_dir, exist_ok=True)

# Get list of tables
with conn.cursor() as cursor:
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()

# Export each table
for (table_name,) in tables:
    print(f"Exporting table: {table_name}")

    # Export data to CSV
    df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    csv_path = os.path.join(csv_dir, f"{table_name}.csv")
    df.to_csv(csv_path, index=False)

    # Export CREATE TABLE script
    with conn.cursor() as cursor:
        cursor.execute(f"SHOW CREATE TABLE {table_name};")
        result = cursor.fetchone()
        create_table_sql = result[1]

        sql_path = os.path.join(sql_dir, f"{table_name}.sql")
        with open(sql_path, 'w') as f:
            f.write(create_table_sql + ";\n")

print(f"\nAll tables exported to '{csv_dir}' (CSV files) and '{sql_dir}' (CREATE scripts).")

# Close connection
conn.close()