import json
import os


class ModelScorer:
    def __init__(self, path: str) -> None:
        self.pred_by_ts = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self.pred_by_ts[int(item["timestamp_ms"])] = item
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

    def get(self, timestamp_ms: int):
        return self.pred_by_ts.get(timestamp_ms)
