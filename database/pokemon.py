import pandas as pd
import pymysql
from db_configs import get_db_connection

CSV_PATH = "pokemon_200.csv"
DB_NAME = "pokemon"
TABLE_NAME = "pokemon_data"

# Step 1: 连接到 MySQL（先不指定数据库，用于创建数据库）
conn = pymysql.connect(
    host="",
    port=3306,
    user="",
    password="",
    charset="utf8mb4",
)
cursor = conn.cursor()

# Step 2: 创建数据库（如果不存在）
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4;")
conn.commit()

# 关闭初始化连接
cursor.close()
conn.close()

# Step 3: 连接到新数据库
conn = get_db_connection(DB_NAME)
cursor = conn.cursor()

# Step 4: 创建表（如果不存在）
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    type_1 VARCHAR(255),
    type_2 VARCHAR(255),
    caption TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
cursor.execute(create_table_sql)
conn.commit()

# Step 5: 读取 CSV
df = pd.read_csv(CSV_PATH)
df = df.where(pd.notnull(df), None)

# Step 6: 插入数据
insert_sql = f"""
INSERT INTO {TABLE_NAME} (name, type_1, type_2, caption)
VALUES (%s, %s, %s, %s)
"""

data_to_insert = df[["name", "type_1", "type_2", "caption"]].values.tolist()

cursor.executemany(insert_sql, data_to_insert)
conn.commit()

cursor.close()
conn.close()

print("✅ 数据库已创建，数据已成功插入！")
