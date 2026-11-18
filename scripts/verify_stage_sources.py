#!/usr/bin/env python3
"""Verify extracted stage source equals the corresponding notebook cell; no experiment execution."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'configs/SOURCE_CELL_MANIFEST.json').read_text(encoding='utf-8'))
notebook=json.loads((ROOT/manifest['notebook']).read_text(encoding='utf-8'))
for entry in manifest['stages']:
    original=''.join(notebook['cells'][entry['cell_index_0_based']]['source']).encode('utf-8')
    extracted=(ROOT/entry['file']).read_bytes()
    assert original==extracted, entry['file']+' differs from notebook source'
    assert hashlib.sha256(extracted).hexdigest()==entry['sha256'],entry['file']+' hash mismatch'
    compile(extracted,str(ROOT/entry['file']),'exec')
    print('PASS:',entry['file'])
print('PASS: all extracted sources match; syntax compiled (no execution)')
