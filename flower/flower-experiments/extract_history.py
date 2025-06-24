"""script for extracting and combining history metrics from Flower experiment logs. Mostly written by chatGPT."""
import sys
import os
import ast
import pandas as pd
import glob
import re
from pathlib import Path


def extract_history_dict(log_path: str) -> dict:
    """
    Scan the log file for the 'History (metrics, centralized):' block and parse it.
    """
    start_token = "History (metrics, centralized):"
    history_lines = []
    capture = False
    
    with open(log_path, "r") as f:
        for line in f:
            # Strip ANSI color codes (e.g. \x1b[92m, \x1b[0m)
            clean = line
            # Start capturing the dict from the first line after the token
            if start_token in clean:
                print(f"Found start token for metrics history.")
                capture = True
                continue
            if capture:
                # Once we hit a line that’s just INFO and whitespace after the dict, we stop
                if clean.strip().startswith("INFO") and clean.strip().endswith("}") is False and not clean.strip().startswith("{"):
                    # But if we've already seen a '}', we break
                    if any('}' in l for l in history_lines):
                        break
                history_lines.append(clean)
    
    if not history_lines:
        raise RuntimeError(f"No history block found in {log_path!r}")
    
    # Join, strip off leading INFO and color codes, then parse
    # We expect something like: "{'accuracy': [(0, 0.1), ...], 'loss': [...], ...}"
    raw = "".join(history_lines)
    # Remove any leading "INFO" tags and ANSI escapes
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)          # strip color
    raw = re.sub(r"INFO\s*:\s*", "", raw)             # strip INFO labels
    raw = raw.strip()
    
    # Make sure it’s a valid Python literal
    # It may span multiple lines, so ast.literal_eval can handle it
    history_dict = ast.literal_eval(raw)
    return history_dict


def extract_run_metadata(log_path: str) -> dict:
    """
    Scans the file for a line of the form:
      Job #1 → method=fedavg, supernodes=10, fraction=1, rounds=50, partition=iid
    and returns a dict with those parsed values.
    """
    pattern = re.compile(
        r"Job\s+#(?P<job>\d+)\s+(→|->)?\s*"
        r"method=(?P<method>[^,]+),\s*"
        r"supernodes=(?P<supernodes>\d+),\s*"
        r"fraction=(?P<fraction>[^,]+),\s*"
        r"rounds=(?P<rounds>\d+),\s*"
        r"partition=(?P<partition>\w+)"
    )

    with open(log_path, "r", encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                meta = m.groupdict()
                # Convert numeric fields
                meta["job_index"]   = int(meta.pop("job"))
                meta["supernodes"]  = int(meta["supernodes"])
                # might be float or int
                meta["fraction"]    = float(meta["fraction"])
                meta["rounds"]      = int(meta["rounds"])
                return meta
    raise RuntimeError(f"No metadata line found in {log_path!r}")


def history_to_dataframe(history: dict, metadata: dict) -> pd.DataFrame:
    """
    Turn the history dict into a DataFrame, and add the metadata as columns.
    """
    # Base DF for the per-round metrics
    rounds = [r for r,_ in history[next(iter(history))]]
    df = pd.DataFrame({"round": rounds})
    for metric, values in history.items():
        df[metric] = [v for _,v in values]
    # Add metadata columns (broadcasted)
    for key, val in metadata.items():
        df[key] = val
    return df


def main():

    log_dir = sys.argv[1]
    job_id = sys.argv[2]

    err_files = list(Path(log_dir).glob("*.err"))
    out_files = list(Path(log_dir).glob("*.out"))


    # filter out unwanted files
    filter_fn = lambda x: job_id in os.path.basename(x)
    err_files = list(filter(filter_fn, err_files))
    out_files = list(filter(filter_fn, out_files))

    # sort files by job index
    key_fn = lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    err_files.sort(key=key_fn)
    out_files.sort(key=key_fn)

    dfs = []
    for i, (err_path, out_path) in enumerate(zip(err_files, out_files)):

        try:
            history = extract_history_dict(err_path)
            metadata = extract_run_metadata(out_path)
            df = history_to_dataframe(history, metadata)
            dfs.append(df)
        except Exception as e:
            print(f"Error processing logs from run {i+1}: {e}")

    if dfs:
        # Concatenate all DataFrames and save to a single CSV
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = Path(out_files[0]).parent / f"{job_id}_combined_history.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Combined history saved to {output_path}")
    

if __name__ == "__main__":
    main()
