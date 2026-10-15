import os
import json
import requests
import hashlib
from pathlib import Path
from ruamel.yaml import YAML

def compute_sha256(file_path):
    hash_func = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def infer_resource_ids_from_maps(manifest_path):
    maps = load_config_file(manifest_path)
    return set(map(lambda m: m['id'].rstrip("abcdefghijklmnopqrstuvwxyz"), maps))

def load_config_file(config_path, return_dict=False):
    path = Path(config_path)
    ext = path.suffix.lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif ext in {".yaml", ".yml"}:
        yaml = YAML(typ="safe")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.load(f)
    else:
        raise ValueError(f"Unsupported manifest file extension: {ext}")

    # Convert list of maps into dict keyed by map ID
    if isinstance(raw, list) and return_dict is True:
        return {entry["id"]: entry for entry in raw}
    return raw


def download_file(url, out_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0'}

    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main(map_manifest_path, resource_manifest_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    resources = load_config_file(resource_manifest_path)

    needed_ids = infer_resource_ids_from_maps(map_manifest_path)
    error_ids = []
    failed_ids = []

    for res_id in sorted(needed_ids):
        if res_id not in resources:
            print(f"[!] WARNING: No entry for resource '{res_id}' in resources.json")
            continue

        info = resources[res_id]
        url = info["url"]
        expected_hash = info.get("sha256")
        out_path = os.path.join(output_dir, f"{res_id}.pdf")

        if os.path.exists(out_path) and expected_hash:
            actual_hash = compute_sha256(out_path)
            if actual_hash == expected_hash:
                print(f"[✓] {res_id}.pdf already exists and hash verified.")
                continue
            else:
                print(f"[!] Hash mismatch for {res_id}, redownloading...")

        print(f"[↓] Downloading {res_id}.pdf...")
        try:
            download_file(url, out_path)
        except requests.RequestException as e:
            print(f"[❌] ERROR: Could not download {res_id}.pdf from {url}: {e}")
            failed_ids.append(res_id)
            continue

        if expected_hash:
            actual_hash = compute_sha256(out_path)
            if actual_hash != expected_hash:
                print(f"[❌] ERROR: Hash mismatch for {res_id}.pdf after download!")
                print(f"Expected: {expected_hash}")
                print(f"Actual:   {actual_hash}")
                error_ids.append(res_id)
            else:
                print(f"[✓] Verified hash for {res_id}.pdf")

    if len(failed_ids) > 0:
        print(f"Warning: {len(failed_ids)} file{'s' if len(failed_ids) > 1 else ''} could not be downloaded! The download links might not work anymore, please let us know.")
        print(f"Maps with failed downloads: {failed_ids}")
    if len(error_ids) > 0:
        print(f"Warning: {len(error_ids)} file{'s' if len(error_ids) > 1 else ''} did not match with the expected hash values! The files provided in the download links might have been changed, so please check them manually.")
        print(f"Maps with errors: {error_ids}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='maps_manifest.yaml')
    parser.add_argument('--resource_manifest', type=str, default='resources.json')
    parser.add_argument('--output_dir', type=str, default='originals')
    args = parser.parse_args()

    main(args.manifest, args.resource_manifest, args.output_dir)
