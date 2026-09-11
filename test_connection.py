"""
智能简历推荐系统 —— 服务连通性验证脚本

逐项验证 MySQL / MongoDB / Milvus / Elasticsearch 四个服务是否可用。
注意：本脚本为「只读」验证，不会创建任何数据库或集合。
"""

import sys

import pymysql
from elasticsearch import Elasticsearch
from pymilvus import MilvusClient
from pymongo import MongoClient

# ============ 连接配置（必须与 docker-compose.yml 保持一致）============

MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DB = "smartrecruit"

MONGO_HOST = "127.0.0.1"
MONGO_PORT = 27017
MONGO_USER = "root"
MONGO_PASSWORD = "123456"

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530

ES_HOST = "127.0.0.1"
ES_PORT = 9200


def verify_mysql():
    """验证 MySQL：能连上 + 有权限查询"""
    print("\n--- 1/4 验证 MySQL ---")
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset="utf8mb4",
        )
        try:
            # 连上不代表有读权限，真正查一次才算验证通过
            with conn.cursor() as cursor:
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()[0]
            print(f"MySQL 连接成功！版本: {version}，库: {MYSQL_DB}")
            return True
        finally:
            # 无论成功失败都要关，否则连接会泄漏
            conn.close()
    except Exception as e:
        print(f"MySQL 连接失败: {type(e).__name__}: {e}")
        return False


def verify_mongo():
    """验证 MongoDB：ping 通即可（库是首次写入数据才创建，此处不检查库）"""
    print("\n--- 2/4 验证 MongoDB ---")
    try:
        client = MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            username=MONGO_USER,
            password=MONGO_PASSWORD,
            # 默认 30s，服务没起会卡很久；缩短到 5s 快速失败
            serverSelectionTimeoutMS=5000,
        )
        try:
            client.admin.command("ping")
            print("MongoDB 连接成功！")
            return True
        finally:
            client.close()
    except Exception as e:
        print(f"MongoDB 连接失败: {type(e).__name__}: {e}")
        return False


def verify_milvus():
    """验证 Milvus：只读取数据库与集合列表，不做任何写操作"""
    print("\n--- 3/4 验证 Milvus ---")
    try:
        # uri 必须带协议头；MilvusClient 没有 port 形参，端口要写进 uri
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        databases = client.list_databases()
        collections = client.list_collections()
        print(f"Milvus 连接成功！已有数据库: {databases}，当前集合: {collections}")
        return True
    except Exception as e:
        print(f"Milvus 连接失败: {type(e).__name__}: {e}")
        return False


def verify_es():
    """验证 Elasticsearch：ping 通后取集群信息"""
    print("\n--- 4/4 验证 Elasticsearch ---")
    try:
        # hosts 的每一项都必须带协议头
        es_client = Elasticsearch(hosts=[f"http://{ES_HOST}:{ES_PORT}"])
        if not es_client.ping():
            print("Elasticsearch ping 失败：服务无响应")
            return False
        info = es_client.info()
        print(
            f"Elasticsearch 连接成功！集群: {info['cluster_name']}，"
            f"版本: {info['version']['number']}"
        )
        return True
    except Exception as e:
        print(f"Elasticsearch 连接失败: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("开始验证所有服务连接...")

    results = [
        ("MySQL", verify_mysql()),
        ("MongoDB", verify_mongo()),
        ("Milvus", verify_milvus()),
        ("Elasticsearch", verify_es()),
    ]

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    print(f"\n===== 验证结果: 通过 {passed}/{total} =====")
    for name, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")

    # 有失败项时返回非 0 退出码，便于脚本/CI 判断
    sys.exit(0 if passed == total else 1)
