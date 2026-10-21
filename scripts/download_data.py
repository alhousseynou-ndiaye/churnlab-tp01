
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"

expected_sha256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

data_path = Path("data/raw/telco.csv")
metadata_path = Path("data/raw/telco.source.json")

if data_path.exists():
    content = data_path.read_bytes()
else:
    with urlopen(url, timeout=30) as response:
        content = response.read()

actual_sha256 = hashlib.sha256(content).hexdigest()

if actual_sha256 != expected_sha256:
    raise ValueError("Dataset SHA-256 mismatch")

data_path.parent.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    data_path.write_bytes(content)

metadata = {
    "source_url": url,
    "sha256": actual_sha256,
    "file": data_path.as_posix(),
}

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print(json.dumps(metadata, indent=2))
