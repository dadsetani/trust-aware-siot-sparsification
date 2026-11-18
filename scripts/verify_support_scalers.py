#!/usr/bin/env python3
"""Read-only cryptographic verification; does not deserialize joblib files."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'docs/SUPPORT_EXPORT_MANIFEST.json').read_text())
records=[r for r in manifest['files'] if r['archive_path'].endswith('.joblib')]
assert len(records)==2
for r in records:
    p=root/'artifacts'/r['archive_path']
    assert p.is_file() and p.stat().st_size==r['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    print('OK:',p.name)
print('Both scaler files match their Kaggle export SHA-256 hashes.')
