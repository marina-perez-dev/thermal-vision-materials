import sys
import csv
import os

def main(in_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # conversion et nettoyage simple
            try:
                temp = float(r.get("temp", r.get("temp_c", 0)))
            except:
                temp = 0.0
            rows.append({"id": r.get("id"), "temp_c": round(temp, 2)})

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id","temp_c"])
        for r in rows:
            writer.writerow([r["id"], r["temp_c"]])

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data_raw.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/data_processed.csv"
    main(inp, out)
