"""
Rerun stage 1 records that failed to produce parseable output.

If no --sites given, runs on all sites.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    VLLM_API_BASE, VLLM_API_KEY, MODEL_NAME,
    STAGE1, SITES, ensure_dirs,
    ID_COL, NOTE_COL, SH_COL,
)
from prompts import build_stage1_prompt
from utils import extract_segments, safe_parse

# Relaxed settings for the rerun: higher token budget, reasoning=low if supported.
RERUN_MAX_OUTPUT_TOKENS = 512
RERUN_REASONING_EFFORT  = 'low'


# =============================================================================
# Identify failed uids in a stage 1 jsonl
# =============================================================================
def identify_failed_uids(jsonl_path: Path) -> list[str]:
    failed = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            uid = rec.get(ID_COL)
            if not uid:
                continue
            if 'error' in rec:
                failed.append(uid)
                continue
            outputs = rec.get('output_text', [])
            if not outputs or all(not s for s in outputs):
                failed.append(uid)
                continue
            parsed = [safe_parse(t) for t in outputs]
            parsed = [p for p in parsed if p and 'self-harm-related' in p]
            if not parsed:
                failed.append(uid)
    return failed


# =============================================================================
# Async inference with relaxed knobs
# =============================================================================
async def infer_one_sample(client, sem, sys_msg, usr_msg):
    async with sem:
        kwargs = dict(
            model=MODEL_NAME,
            instructions=sys_msg,
            input=usr_msg,
            temperature=STAGE1['temperature'],
            top_p=STAGE1['top_p'],
            max_output_tokens=RERUN_MAX_OUTPUT_TOKENS,
        )
        if RERUN_REASONING_EFFORT:
            kwargs['reasoning'] = {'effort': RERUN_REASONING_EFFORT}
        try:
            resp = await client.responses.create(**kwargs)
        except TypeError:
            kwargs.pop('reasoning', None)
            resp = await client.responses.create(**kwargs)
    return extract_segments(resp)


async def infer_one_row(client, sem, uid, sh, note):
    sys_msg, usr_msg = build_stage1_prompt(note)
    tasks = [
        infer_one_sample(client, sem, sys_msg, usr_msg)
        for _ in range(STAGE1['n_runs'])
    ]
    samples = await asyncio.gather(*tasks)
    return {
        ID_COL: uid,
        SH_COL: int(sh) if sh is not None else None,
        'reasoning_text': [r for r, _ in samples],
        'output_text':    [o for _, o in samples],
    }


# =============================================================================
# Per-site rerun
# =============================================================================
async def rerun_site(client, sem, site_key: str):
    site = SITES[site_key]
    parquet_path = site['parquet']
    jsonl_path   = site['stage1_jsonl']

    print('=' * 70)
    print(f'Site: {site_key}')
    print(f'  parquet: {parquet_path}')
    print(f'  jsonl:   {jsonl_path}')
    print('=' * 70)

    if not jsonl_path.exists():
        print('  Stage 1 jsonl missing; nothing to redo.')
        return

    failed = identify_failed_uids(jsonl_path)
    print(f'  Found {len(failed):,} failed records.')
    if not failed:
        return

    df = pd.read_parquet(parquet_path)[[ID_COL, SH_COL, NOTE_COL]].dropna(subset=[NOTE_COL])
    df = df[df[ID_COL].isin(failed)].reset_index(drop=True)
    print(f'  Matched {len(df):,} rows in parquet to rerun.')
    if len(df) == 0:
        return

    new_records = {}
    tasks = [
        infer_one_row(client, sem, row[ID_COL], row[SH_COL], str(row[NOTE_COL]))
        for _, row in df.iterrows()
    ]
    done = 0
    for coro in asyncio.as_completed(tasks):
        try:
            rec = await coro
            new_records[rec[ID_COL]] = rec
        except Exception as e:
            print(f'  [worker error] {repr(e)[:200]}')
        done += 1
        if done % 100 == 0:
            print(f'  progress: {done}/{len(tasks)}')

    # Backup and rewrite the jsonl
    backup = jsonl_path.with_suffix(jsonl_path.suffix + '.bak_rerun')
    shutil.copy(jsonl_path, backup)
    print(f'  Backed up original to {backup}')

    out_lines, replaced = [], 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                out_lines.append(line)
                continue
            uid = rec.get(ID_COL)
            if uid in new_records:
                out_lines.append(json.dumps(new_records[uid], ensure_ascii=False) + '\n')
                replaced += 1
            else:
                out_lines.append(line)
    with open(jsonl_path, 'w') as f:
        f.writelines(out_lines)
    print(f'  Replaced {replaced:,} records.')


async def main_async(site_keys):
    ensure_dirs()
    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(STAGE1['max_in_flight'])
    for site_key in site_keys:
        await rerun_site(client, sem, site_key)


def parse_args():
    p = argparse.ArgumentParser(description='Rerun failed Stage 1 records')
    p.add_argument('--sites', nargs='+', default=None,
                   help='Site keys from config.SITES to rerun. Default: all.')
    return p.parse_args()


def main():
    args = parse_args()
    site_keys = args.sites if args.sites else list(SITES.keys())
    unknown = [k for k in site_keys if k not in SITES]
    if unknown:
        raise SystemExit(f'Unknown site keys: {unknown}. '
                         f'Available: {list(SITES.keys())}')
    asyncio.run(main_async(site_keys))


if __name__ == '__main__':
    main()
