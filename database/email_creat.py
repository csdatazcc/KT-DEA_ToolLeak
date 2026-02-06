import pandas as pd
from db_configs import get_db_connection

CSV_PATH = "marketing_emails_200.csv"  # 你的 CSV 文件
DB_NAME = ""               # 使用同一个数据库
TABLE_NAME = ""

# Step 1: 连接数据库
conn = get_db_connection(DB_NAME)
cursor = conn.cursor()

# Step 2: 创建表（如果不存在）
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject VARCHAR(255),
    `from` VARCHAR(255),
    `to` VARCHAR(255),
    context TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
cursor.execute(create_table_sql)
conn.commit()

# Step 3: 读取 CSV
df = pd.read_csv(CSV_PATH)

# Step 4: 将 NaN 转成 None
df = df.where(pd.notnull(df), None)

# Step 5: 插入数据
insert_sql = f"""
INSERT INTO {TABLE_NAME} (subject, `from`, `to`, context)
VALUES (%s, %s, %s, %s)
"""

data_to_insert = df[["subject", "from", "to", "context"]].values.tolist()
cursor.executemany(insert_sql, data_to_insert)
conn.commit()

cursor.close()
conn.close()

print("✅ marketing_email 数据已成功导入数据库")
