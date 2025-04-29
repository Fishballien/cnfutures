# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:38:13 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pymysql
import yaml
import time
from pathlib import Path
from datetime import datetime


from utils.logutils import FishStyleLogger
from utils.dirutils import load_path_config


# %%
class DatabaseHandler:
    
    def __init__(self, mysql_name, log=None):
        self.mysql_name = mysql_name
        self.log = log
        
        self._init_logger()
        self._load_path_config()
        self._load_sql_config()
        
    def _init_logger(self):
        """初始化日志"""
        self.log = self.log or FishStyleLogger()
            
    def _load_path_config(self):
        """加载路径配置"""
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        path_config = load_path_config(project_dir)
        
        self.sql_config_dir = Path(path_config['sql_config'])
        
    def _load_sql_config(self):
        """加载 SQL 配置"""
        file_path = self.sql_config_dir / f'{self.mysql_name}.yaml'
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        self.mysql_info = config['mysql']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay']
        
    def connect(self):
        """尝试建立数据库连接，最多重试 max_retries 次"""
        retries = 0
        connection = None
        while retries < self.max_retries:
            try:
                # 尝试建立数据库连接
                connection = pymysql.connect(**self.mysql_info)
                if connection.open:
                    return connection
            except pymysql.MySQLError as e:
                retries += 1
                self.log.warning(f"Connection attempt {retries} failed: {e}")
                if retries < self.max_retries:
                    self.log.warning(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.log.error("Max retries reached, failed to connect to the database")
                    raise e  # 超过重试次数，抛出异常
        return connection


# %% daily update sender
class DailyUpdateSender(DatabaseHandler):
    
    def __init__(self, mysql_name, author, log=None):
        """
        初始化 DailyUpdateSender，设置数据库连接和作者名称
        :param mysql_name: 数据库连接名称
        :param author: 数据作者
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
        self.author = author  # 初始化 author
    
    def insert(self, obj, data_ts, status=1):
        """
        插入一条记录到 daily_update 表。
        :param obj: 数据对象
        :param data_ts: 数据时间戳
        :param status: 状态
        """
        connection = None
        cursor = None
        
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return  # 无法连接数据库，停止执行
            
            # 准备插入的 SQL 语句
            insert_query = """
            INSERT INTO daily_update (author, object, data_ts, status)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE data_ts = VALUES(data_ts), 
                                    status = VALUES(status), 
                                    timestamp = CURRENT_TIMESTAMP;
            """
            
            # 执行插入操作
            cursor = connection.cursor()
            cursor.execute(insert_query, (self.author, obj, data_ts, status))
            connection.commit()  # 提交事务
            self.log.success(f"Successfully inserted record for object '{obj}' into daily_update.")
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while inserting into daily_update: {e}")
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()


# %% daily update reader
class DailyUpdateReader(DatabaseHandler):
    
    def __init__(self, mysql_name, log=None):
        """
        初始化 DailyUpdateReader，设置数据库连接
        :param mysql_name: 数据库连接名称
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
    
    def fetch(self, author, object_name, data_ts):
        """
        根据作者、对象名称和指定的 data_ts 获取 daily_update 数据，
        返回所有大于等于指定 data_ts 的 (data_ts, status) 列表。
        :param author: 数据作者
        :param object_name: 数据对象名称
        :param data_ts: 最早的 data_ts 时间
        :return: [(data_ts, status), ...] 或空列表
        """
        connection = None
        cursor = None
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return []  # 无法连接数据库，返回空列表
            
            # 查询 SQL 语句
            select_query = """
            SELECT data_ts, status
            FROM daily_update
            WHERE author = %s AND object = %s AND data_ts >= %s;
            """
            
            # 执行查询操作
            cursor = connection.cursor()
            cursor.execute(select_query, (author, object_name, data_ts))
            results = cursor.fetchall()  # 获取所有符合条件的结果
            
            if results:
                return [(row[0], row[1]) for row in results]  # 返回 (data_ts, status) 列表
            
            return []  # 如果没有结果，返回空列表
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while fetching data from daily_update: {e}")
            return []
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
                
# %% daily update message sender
class DailyUpdateMsgSender(DatabaseHandler):
    
    def __init__(self, mysql_name, author, log=None):
        """
        初始化 DailyUpdateMsgSender，设置数据库连接和作者名称
        :param mysql_name: 数据库连接名称
        :param author: 消息的作者
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
        self.author = author  # 初始化 author
    
    def insert(self, level, title, msg):
        """
        插入一条消息记录到 daily_update_msg 表。
        :param level: 消息的级别
        :param title: 消息标题
        :param msg: 消息内容
        """
        connection = None
        cursor = None
        
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return  # 无法连接数据库，停止执行
            
            # 准备插入的 SQL 语句
            insert_query = """
            INSERT INTO daily_update_msg (author, level, title, msg)
            VALUES (%s, %s, %s, %s);
            """
            
            # 执行插入操作
            cursor = connection.cursor()
            cursor.execute(insert_query, (self.author, level, title, msg))
            connection.commit()  # 提交事务
            self.log.success(f"Successfully inserted message with title '{title}' into daily_update_msg.")
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while inserting into daily_update_msg: {e}")
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
                
# %% daily update message reader
class DailyUpdateMsgReader(DatabaseHandler):
    
    def __init__(self, mysql_name, log=None):
        """
        初始化 DailyUpdateMsgReader，设置数据库连接
        :param mysql_name: 数据库连接名称
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
    
    def read(self, include=None, exclude=None):
        """
        从 daily_update_msg 表中读取满足条件的记录。
        :param include: 包含的 level 列表
        :param exclude: 排除的 level 列表
        :return: 满足条件的记录列表
        """
        connection = None
        cursor = None
        result = []
        
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return result  # 无法连接数据库，返回空列表
            
            # 构造基础查询语句
            base_query = "SELECT * FROM daily_update_msg WHERE alerted = 0"
            query_params = []
            
            # 优先处理 include 条件
            if include:
                placeholders = ', '.join(['%s'] * len(include))
                base_query += f" AND level IN ({placeholders})"
                query_params.extend(include)
            # 如果 include 为空，处理 exclude 条件
            elif exclude:
                placeholders = ', '.join(['%s'] * len(exclude))
                base_query += f" AND level NOT IN ({placeholders})"
                query_params.extend(exclude)
            
            # 执行查询操作
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute(base_query, query_params)
            
            # 获取查询结果
            result = cursor.fetchall()
            self.log.success("Successfully read messages from daily_update_msg.")
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while reading from daily_update_msg: {e}")
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        
        return result


# %% daily update message updater
class DailyUpdateMsgUpdater(DatabaseHandler):
    
    def __init__(self, mysql_name, log=None):
        """
        初始化 DailyUpdateMsgUpdater，设置数据库连接
        :param mysql_name: 数据库连接名称
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
    
    def update_alerted(self, id_list):
        """
        将指定 id 的行的 alerted 字段更新为 1。
        :param id_list: 要更新的 id 列表
        """
        if not id_list:
            self.log.warning("No IDs provided for update. Operation aborted.")
            return
        
        connection = None
        cursor = None
        
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return  # 无法连接数据库，停止执行
            
            # 构造更新语句
            placeholders = ', '.join(['%s'] * len(id_list))
            update_query = f"""
            UPDATE daily_update_msg
            SET alerted = 1
            WHERE id IN ({placeholders});
            """
            
            # 执行更新操作
            cursor = connection.cursor()
            cursor.execute(update_query, id_list)
            connection.commit()  # 提交事务
            self.log.success(f"Successfully updated {cursor.rowcount} rows in daily_update_msg.")
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while updating daily_update_msg: {e}")
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                
                
# %%
class PythonTradeBacktestHandler(DatabaseHandler):
    
    def __init__(self, mysql_name, log=None):
        """
        初始化 PythonTradeBacktestHandler，设置数据库连接和作者名称
        :param mysql_name: 数据库连接名称
        :param log: 日志对象（可选）
        """
        super().__init__(mysql_name, log=log)
    
    def insert(self, df, acname):
        """
        将 DataFrame 中的数据插入到 python_trade_backtest 表。
        
        :param df: 包含 symbol 为列，index 为时间戳（stockdate），值为 position 的 DataFrame
        :param acname: 账户名称
        """
        connection = None
        cursor = None
        
        try:
            # 建立数据库连接
            connection = self.connect()
            if not connection:
                return  # 无法连接数据库，停止执行
            
            # 构造 SQL 插入语句
            insert_query = """
            INSERT INTO python_trade_backtest (acname, stockdate, symbol, position, inserttime)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE position = VALUES(position), inserttime = VALUES(inserttime);
            """
            
            # 将 DataFrame 转换为适合插入的数据格式
            values = []
            for timestamp, row in df.iterrows():
                for symbol, position in row.items():
                    # 每一行对应一个时间戳和symbol的值
                    values.append((acname, timestamp, symbol, position, datetime.now()))
            
            # 执行批量插入操作
            cursor = connection.cursor()
            cursor.executemany(insert_query, values)
            connection.commit()  # 提交事务
            self.log.success(f"Successfully inserted {len(values)} rows into python_trade_backtest.")
        
        except pymysql.MySQLError as e:
            self.log.error(f"Error while inserting into python_trade_backtest: {e}")
        
        finally:
            # 关闭游标和连接
            if cursor:
                cursor.close()
            if connection:
                connection.close()

