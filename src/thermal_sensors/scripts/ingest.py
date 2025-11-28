import sys
import os
import csv

def main(src_dir, out_path):
    # src_dir peut être un dossier de source ou laissé vide pour générer un sample
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = [["id","temp"]]
    if os.path.isdir(src_dir) and any(os.scandir(src_dir)):
        # exemple : concatener tous les CSV du dossier src_dir
        for entry in os.scandir(src_dir):
            if entry.name.lower().endswith(".csv"):
                with open(entry.path, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    rows.extend(row for row in reader)
    else:
        # données d'exemple
        rows.extend([["1","36.5"], ["2","37.2"], ["3","35.9"]])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data_src"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/data_raw.csv"
    main(src, out)
