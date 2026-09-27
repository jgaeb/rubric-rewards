#!/usr/bin/env python
"""Download the replication package's data from the Hugging Face dataset."""
import argparse
import pathlib

from huggingface_hub import snapshot_download

# Define which files each layer of the package downloads: the tables are enough
# to draw the figures, while the raw records are needed to rebuild the tables,
# refit the rewards, and run the check
LAYERS = {
    "tables": ["tables/*", "figures/*"],
    "raw": ["raw/**", "MANIFEST_raw.json"],
    "all": ["raw/**", "tables/*", "figures/*", "MANIFEST_raw.json"],
}


################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="jgaeb/rubric-rewards")
    parser.add_argument("--what", type=str, default="tables", choices=list(LAYERS))
    parser.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args()

    # Download the requested layer into the destination directory
    download_directory = snapshot_download(
        args.repo,
        repo_type="dataset",
        local_dir=args.dest,
        allow_patterns=LAYERS[args.what],
    )
    print(f"Downloaded to {download_directory}")


if __name__ == "__main__":
    main()
