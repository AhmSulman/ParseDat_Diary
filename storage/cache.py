import hashlib, json, os

CACHE_FILE = "data/cache/hashes.json"

class Cache:
    def __init__(self):
        os.makedirs("data/cache", exist_ok=True)
        if not os.path.exists(CACHE_FILE):
            with open(CACHE_FILE,"w") as f: json.dump({},f)
        with open(CACHE_FILE) as f: self.hashes = json.load(f)

    def hash_file(self, filepath):
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""): md5.update(chunk)
        return md5.hexdigest()

    def seen(self, h): return h in self.hashes

    def record(self, h, name):
        self.hashes[h] = name
        with open(CACHE_FILE,"w") as f: json.dump(self.hashes, f, indent=2)
