# db_config.py
import pymysql
from pymysql.cursors import DictCursor

def get_db_connection(database_name: str):
    """
    返回数据库连接（pymysql），根据指定数据库名连接不同数据库。
    """
    return pymysql.connect(
        host="",
        port=3306,
        user="",
        password="",
        database=database_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
