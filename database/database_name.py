import pymysql

OLD_DB = ""   # 这里改成你现在的数据库名字
NEW_DB = ""

# 连接 MySQL（不指定数据库）
conn = pymysql.connect(
    host="",
    port=3306,
    user="",
    password="",
    charset="utf8mb4"
)
cursor = conn.cursor()

# 1. 创建新数据库
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {NEW_DB} CHARACTER SET utf8mb4;")

# 2. 获取原数据库所有表
cursor.execute(f"SHOW TABLES IN {OLD_DB};")
tables = [t[0] for t in cursor.fetchall()]

# 3. 复制表结构和数据
for table in tables:
    # 创建表结构
    cursor.execute(f"CREATE TABLE {NEW_DB}.{table} LIKE {OLD_DB}.{table};")
    # 复制数据
    cursor.execute(f"INSERT INTO {NEW_DB}.{table} SELECT * FROM {OLD_DB}.{table};")

conn.commit()
cursor.close()
conn.close()

print(f"✅ 数据已迁移到新数据库: {NEW_DB}")
