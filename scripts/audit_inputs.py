#!/usr/bin/env python3
"""Read-only audit of all frozen checkpoint paths, sizes, and SHA-256 hashes."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
protocol=json.loads((ROOT/'configs/FROZEN_PROTOCOL_BEFORE_TEST.json').read_text())
man=json.loads((ROOT/'artifacts/FROZEN_CHECKPOINT_EXPORT_MANIFEST.json').read_text())
assert set(protocol['checkpoint_paths'])=={r['source'] for r in man['files']},'Frozen protocol paths mismatch'
errors=[]
for record in man['files']:
    file=ROOT/'artifacts'/record['archive_path']
    if not file.is_file():errors.append(('MISSING',str(file)));continue
    h=hashlib.sha256()
    with file.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if file.stat().st_size != record['bytes'] or h.hexdigest()!=record['sha256']:
        errors.append(('MISMATCH',str(file)))
print('Frozen checkpoint records:',len(man['files']))
print('Verified:',len(man['files'])-len(errors))
print('Errors:',len(errors))
for kind,path in errors:print(kind,path)
print('Portable training or held-out inference: NOT validated')
if errors:raise SystemExit(1)
