"""Fail closed when prior GPU cost or a physical teacher call is uncertain."""
import json,os

def begin_once(path, record):
    try:
        with path.open('x') as f:
            json.dump(record,f); f.flush(); os.fsync(f.fileno())
    except FileExistsError as exc:
        raise RuntimeError('prior invocation exists; reconcile GPU use and attempts before restarting') from exc

def validate_selection(pool,arms):
    if len(pool)!=50 or len(set(pool))!=50: raise ValueError('invalid candidate pool')
    for a,ids in arms.items():
        if len(ids)!=10 or len(set(ids))!=10 or not set(ids)<=set(pool): raise ValueError('invalid selected IDs: '+a)
