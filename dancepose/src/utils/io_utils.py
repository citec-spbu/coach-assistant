import os, json, time
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

class JsonlWriter:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.count = 0
    def write(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.count += 1
    def close(self):
        self.f.close()

class SimpleLogger:
    def __init__(self, path):
        self.path = path
        self.lines = []
    def log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.lines.append(line)
    def flush(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
