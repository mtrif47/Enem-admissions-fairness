"""
Sample rows from the ENEM microdata CSV without loading the full file into memory.

Usage:
    python3 sample_enem.py


"""

import pandas as pd
import os


INPUT_PATH = "MICRODADOS_ENEM_2019.csv"   # path to the big file inside DADOS/
OUTPUT_PATH = "enem_2019_sample.csv"
SAMPLE_FRACTION = 0.04                     # ~4% -> roughly 200k rows if the file has ~5M
CHUNK_SIZE = 200_000
RANDOM_STATE = 42
# ------------------

def detect_delimiter_and_encoding(path):
    """INEP files have used ';' delimiters and either UTF-8 or Latin-1 encoding
    depending on the year. Try the combinations INEP has historically used."""
    candidates = [
        (";", "utf-8"),
        (";", "latin-1"),
        (",", "utf-8"),
        (",", "latin-1"),
    ]
    for sep, enc in candidates:
        try:
            test = pd.read_csv(path, sep=sep, encoding=enc, nrows=5)
            if test.shape[1] > 1:  # more than one column means the delimiter is right
                print(f"Detected: sep='{sep}', encoding='{enc}'")
                print(f"Columns found ({test.shape[1]}): {list(test.columns)[:10]}...")
                return sep, enc
        except Exception:
            continue
    raise ValueError("Could not auto-detect delimiter/encoding. Open the file "
                      "in a text editor first and check manually.")


def sample_large_csv(input_path, output_path, frac, chunk_size, sep, encoding, seed):
    first_chunk = True
    total_rows_seen = 0
    total_rows_sampled = 0

    reader = pd.read_csv(
        input_path,
        sep=sep,
        encoding=encoding,
        chunksize=chunk_size,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)
        sampled = chunk.sample(frac=frac, random_state=seed + i)  # vary seed per chunk
        total_rows_sampled += len(sampled)

        sampled.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            sep=sep,
            encoding=encoding,
        )
        first_chunk = False

        if i % 5 == 0:
            print(f"  ...processed {total_rows_seen:,} rows so far, "
                  f"sampled {total_rows_sampled:,}")

    print(f"\nDone. Total rows in source: {total_rows_seen:,}")
    print(f"Total rows sampled: {total_rows_sampled:,}")
    print(f"Sample written to: {output_path}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Edit INPUT_PATH at the top of this "
            f"script to point at your actual file location."
        )

    print(f"Inspecting {INPUT_PATH} ...")
    sep, encoding = ";", "latin-1"

    print(f"\nSampling ~{SAMPLE_FRACTION:.0%} of rows in chunks of {CHUNK_SIZE:,} ...")
    sample_large_csv(INPUT_PATH, OUTPUT_PATH, SAMPLE_FRACTION, CHUNK_SIZE,
                      sep, encoding, RANDOM_STATE)
