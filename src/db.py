# src/db.py
from pymongo import MongoClient

# Conexion con atlas
MONGO_URI = "mongodb+srv://usuariobio:Mariana2026@cluster0.sv9scew.mongodb.net/?appName=Cluster0"

def get_db(uri=MONGO_URI, db_name="biooptimization"):
    client = MongoClient(uri)
    db = client[db_name]
    return db["runs"]


def save_run(collection, run_doc):
    result = collection.insert_one(run_doc)
    return result.inserted_id


def best_run(collection, algorithm):
    return collection.find_one(
        {"algorithm": algorithm},
        sort=[("metrics.mse", 1)]
    )


def avg_mse_by_algorithm(collection):
    pipeline = [
        {"$group": {
            "_id": "$algorithm",
            "avg_mse": {"$avg": "$metrics.mse"},
            "std_mse": {"$stdDevPop": "$metrics.mse"},
            "n": {"$sum": 1}
        }},
        {"$sort": {"avg_mse": 1}}
    ]
    return list(collection.aggregate(pipeline))