import json, os
from logs.logger import log

STATE_FILE = "data/checkpoints/state.json"

class Checkpoint:
    def __init__(self):
        os.makedirs("data/checkpoints", exist_ok=True)
        if not os.path.exists(STATE_FILE):
            self._write({"done": [], "failed": []})
        self.state = self._read()

    def is_done(self, pdf): return pdf in self.state["done"]
    def is_failed(self, pdf): return pdf in self.state["failed"]

    def mark_done(self, pdf):
        if pdf not in self.state["done"]: self.state["done"].append(pdf)
        if pdf in self.state["failed"]: self.state["failed"].remove(pdf)
        self._write(self.state)

    def mark_failed(self, pdf):
        if pdf not in self.state["failed"]: self.state["failed"].append(pdf)
        self._write(self.state)

    def reset(self):
        self._write({"done": [], "failed": []})
        self.state = {"done": [], "failed": []}
        log.info("Checkpoints reset")

    def _read(self):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: return {"done": [], "failed": []}

    def _write(self, data):
        with open(STATE_FILE, "w") as f: json.dump(data, f, indent=2)
