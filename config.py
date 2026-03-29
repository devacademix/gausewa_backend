import json
import os
import threading
import uuid
from pathlib import Path

from pymongo import MongoClient


class LocalInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class LocalUpdateResult:
    def __init__(self, matched_count: int):
        self.matched_count = matched_count


class LocalCursor:
    def __init__(self, docs):
        self._docs = docs

    def __iter__(self):
        return iter(self._docs)

    def limit(self, n: int):
        return LocalCursor(self._docs[:n])


class LocalCollection:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def _read_docs(self):
        try:
            raw = self.storage_path.read_text(encoding="utf-8")
            docs = json.loads(raw) if raw.strip() else []
            return docs if isinstance(docs, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_docs(self, docs):
        self.storage_path.write_text(
            json.dumps(docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _matches(doc, query):
        if not query:
            return True
        for key, expected in query.items():
            if isinstance(expected, dict) and "$exists" in expected:
                exists = key in doc and doc[key] is not None
                if exists != bool(expected["$exists"]):
                    return False
            elif doc.get(key) != expected:
                return False
        return True

    @staticmethod
    def _project(doc, projection):
        if not projection:
            return dict(doc)

        include = [k for k, v in projection.items() if v == 1 and k != "_id"]
        if include:
            return {k: doc[k] for k in include if k in doc}

        excluded = {k for k, v in projection.items() if v == 0}
        return {k: v for k, v in doc.items() if k not in excluded}

    def find(self, query=None, projection=None):
        with self._lock:
            docs = self._read_docs()
            out = [
                self._project(doc, projection)
                for doc in docs
                if self._matches(doc, query)
            ]
            return LocalCursor(out)

    def find_one(self, query=None, projection=None):
        with self._lock:
            docs = self._read_docs()
            for doc in docs:
                if self._matches(doc, query):
                    return self._project(doc, projection)
            return None

    def insert_one(self, doc):
        with self._lock:
            docs = self._read_docs()
            new_doc = dict(doc)
            new_doc.setdefault("_id", str(uuid.uuid4()))
            docs.append(new_doc)
            self._write_docs(docs)
            return LocalInsertResult(new_doc["_id"])

    def update_one(self, query, update):
        with self._lock:
            docs = self._read_docs()
            matched = 0
            for idx, doc in enumerate(docs):
                if not self._matches(doc, query):
                    continue

                if "$set" in update:
                    for key, value in update["$set"].items():
                        doc[key] = value

                if "$push" in update:
                    for key, value in update["$push"].items():
                        existing = doc.get(key)
                        if isinstance(existing, list):
                            existing.append(value)
                        else:
                            doc[key] = [value]

                docs[idx] = doc
                matched = 1
                break

            if matched:
                self._write_docs(docs)
            return LocalUpdateResult(matched)

    def count_documents(self, query=None):
        with self._lock:
            docs = self._read_docs()
            return sum(1 for doc in docs if self._matches(doc, query))

    def aggregate(self, pipeline):
        with self._lock:
            docs = self._read_docs()
            if (
                len(pipeline) == 1
                and "$group" in pipeline[0]
                and pipeline[0]["$group"].get("count") == {"$sum": 1}
            ):
                key_expr = pipeline[0]["$group"].get("_id")
                if isinstance(key_expr, str) and key_expr.startswith("$"):
                    key_name = key_expr[1:]
                    grouped = {}
                    for doc in docs:
                        key = doc.get(key_name)
                        grouped[key] = grouped.get(key, 0) + 1
                    return [{"_id": k, "count": v} for k, v in grouped.items()]
            return []


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://ashiepeter01_db_user:UgeQPRCWPiW6SfHk@cluster0.mauq9kc.mongodb.net/?appName=Cluster0",
    
)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "gausewa")
LOCAL_COWS_FILE = os.getenv(
    "LOCAL_COWS_FILE",
    str(Path(__file__).resolve().parent / "dataset" / "cows_local.json"),
)
USE_LOCAL_DB = os.getenv("USE_LOCAL_DB", "0").strip() == "1"

client = None
db = None
DB_BACKEND = "local"

if not USE_LOCAL_DB:
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True,
        )
        client.admin.command("ping")
        db = client[MONGO_DB_NAME]
        cows_col = db["cows"]
        DB_BACKEND = "mongo"
        print("[config] Using MongoDB backend")
    except Exception as exc:
        print(f"[config] MongoDB unavailable ({exc}). Falling back to local file DB.")
        cows_col = LocalCollection(LOCAL_COWS_FILE)
else:
    cows_col = LocalCollection(LOCAL_COWS_FILE)
    print("[config] Using local file DB backend")

JWT_SECRET = os.getenv("JWT_SECRET", "gausewa-secret-change-in-prod")
JWT_EXPIRE = 60
