"""Download the CELLxGENE query datasets named in the manifest.

Acquisition stage. Writes only into ``0_data/queries/cellxgene/``, which is
read-only input by project rule once it lands — nothing downstream ever modifies
these files, and they are gitignored like the rest of ``0_data``.

The manifest beside the files records, per dataset, its id, collection id, title,
cell and donor counts, tissues, diseases, publication date, byte size and source
URL. Provenance is captured at download time rather than reconstructed later,
which is the whole reason a manifest exists.

Resumable and idempotent: a file already present at its manifest size is skipped,
and a partial file is continued with an HTTP range request. Re-running after an
interruption costs only what is missing.

Sizes are verified against the manifest on completion. A truncated download that
is silently accepted would surface much later as a corrupt h5ad, so it fails here
instead.
"""

import csv
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = pathlib.Path(__file__).resolve().parents[2]
DEST = BASE / "0_data/queries/cellxgene"
WORKERS = 4          # the CDN caps per-connection, not total; 4 saturates it
CHUNK = 4 << 20


def fetch(row: dict) -> tuple[str, str]:
    """Download one dataset, resuming if a partial file exists.

    Returns:
        (dataset_id, status) where status is ``skip``, ``ok`` or an error string.
    """
    # One directory per dataset, mirroring 2_results/<DATASET>/ so a dataset
    # is a self-contained unit at every level of the tree.
    d = DEST / row["dataset_id"]
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{row['dataset_id']}.h5ad"
    want = int(row["bytes"])
    have = out.stat().st_size if out.exists() else 0
    if have == want:
        return row["dataset_id"], "skip"
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    try:
        with requests.get(row["url"], headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out, mode) as fh:
                for chunk in r.iter_content(CHUNK):
                    fh.write(chunk)
    except Exception as e:                      # noqa: BLE001 - reported, not hidden
        return row["dataset_id"], f"ERROR {type(e).__name__}: {e}"
    got = out.stat().st_size
    if got != want:
        return row["dataset_id"], f"ERROR size {got:,} != manifest {want:,}"
    return row["dataset_id"], "ok"


def main() -> None:
    """Fetch everything in the manifest that is not already complete."""
    t0 = time.time()
    rows = list(csv.DictReader(open(DEST / "MANIFEST.tsv"), delimiter="\t"))
    total = sum(int(r["bytes"]) for r in rows)
    done_b = sum((DEST / r["dataset_id"] / f"{r['dataset_id']}.h5ad").stat().st_size
                 for r in rows if (DEST / r["dataset_id"] / f"{r['dataset_id']}.h5ad").exists())
    print(f"{len(rows)} datasets, {total/1e9:.1f} GB total, "
          f"{done_b/1e9:.1f} GB already present", flush=True)

    ok = skip = bad = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch, r): r for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            ds, status = f.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                bad += 1
                print(f"  {ds}  {status}", flush=True)
            have = sum((DEST / r["dataset_id"] / f"{r['dataset_id']}.h5ad").stat().st_size
                       for r in rows if (DEST / r["dataset_id"] / f"{r['dataset_id']}.h5ad").exists())
            print(f"[{time.time()-t0:.0f}s] {i}/{len(rows)} files — "
                  f"{have/1e9:5.1f}/{total/1e9:.1f} GB "
                  f"({have/total*100:4.1f}%)  ok={ok} skip={skip} err={bad}",
                  flush=True)

    print(f"\n[{time.time()-t0:.0f}s] DOWNLOADS FINISHED  "
          f"ok={ok} skipped={skip} errors={bad}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
