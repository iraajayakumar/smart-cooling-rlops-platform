import json
import csv
from pathlib import Path

from experiments.utils.writer import write_csv, write_json


def test_write_csv_and_json(tmp_path):
    rows = [
        {"step": 0, "temperature": 35.0, "reward": 1.2},
        {"step": 1, "temperature": 34.5, "reward": 1.5},
    ]

    csv_path = tmp_path / "out.csv"
    json_path = tmp_path / "out.json"

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    assert csv_path.exists()
    assert json_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert len(loaded) == 2
    assert loaded[0]["step"] == "0"

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[1]["reward"] == 1.5