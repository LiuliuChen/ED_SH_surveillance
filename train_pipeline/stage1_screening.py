"""
Stage 1: zero-shot LLM screening.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# Make config.py (one level up) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import *
from prompts import build_stage1_prompt
from utils import extract_segments


# ID_COL / NOTE_COL / SH_COL come from config via the wildcard import above.


"""
Resume mode: read uids already written to the output file
"""
def load_done_uids(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                uid = rec.get(ID_COL)
                if uid is not None:
                    done.add(uid)
            except Exception:
                pass
    return done



"""
Single-call and per-row inference
"""
async def infer_one_sample(client, sem, sys_msg, usr_msg):
    """One LLM call, returns (reasoning_text, output_text) as strings."""
    async with sem:
        kwargs = dict(
            model=MODEL_NAME,
            instructions=sys_msg,
            input=usr_msg,
            temperature=STAGE1['temperature'],
            top_p=STAGE1['top_p'],
            max_output_tokens=STAGE1['max_output_tokens'],
        )
        # Reasoning effort. gpt-oss accepts this; non-reasoning models reject it,
        # so we retry without on TypeError.
        if STAGE1.get('reasoning_effort'):
            kwargs['reasoning'] = {'effort': STAGE1['reasoning_effort']}
        try:
            resp = await client.responses.create(**kwargs)
        except TypeError:
            kwargs.pop('reasoning', None)
            resp = await client.responses.create(**kwargs)
    return extract_segments(resp)


async def infer_one_row(client, sem, uid, sh, note):
    """Run N_RUNS self-consistency samples for one note, return one jsonl record."""
    sys_msg, usr_msg = build_stage1_prompt(note)
    tasks = [
        infer_one_sample(client, sem, sys_msg, usr_msg)
        for _ in range(STAGE1['n_runs'])
    ]
    samples = await asyncio.gather(*tasks)
    return {
        ID_COL:            uid,
        SH_COL:             int(sh) if sh is not None else None,
        'reasoning_text': [r for r, _ in samples],
        'output_text':    [o for _, o in samples],
    }


"""
Per-site runner
"""
async def run_site(client, sem, site_key: str):
    site = SITES[site_key]
    parquet_path: Path = site['parquet']
    out_path:     Path = site['stage1_jsonl']

    print('=' * 70)
    print(f'Site: {site_key}')
    print(f'  parquet: {parquet_path}')
    print(f'  output:  {out_path}')
    print('=' * 70)

    if not parquet_path.exists():
        print(f'  ERROR: parquet not found, skipping site')
        return

    df = pd.read_parquet(parquet_path)[[ID_COL, SH_COL, NOTE_COL]]
    df = df.dropna(subset=[NOTE_COL]).reset_index(drop=True)

    # Resume: drop uids already written
    done = load_done_uids(out_path)
    if done:
        before = len(df)
        df = df[~df[ID_COL].isin(done)].reset_index(drop=True)
        print(f'  resume: {len(done):,} uids already done; '
              f'{before:,} -> {len(df):,} remaining')

    if len(df) == 0:
        print(f'  Nothing to do, moving on.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = STAGE1['chunk_size']

    with open(out_path, 'a', encoding='utf-8', buffering=1, errors='replace') as fout:
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            part = df.iloc[start:end]
            print(f'  chunk {start}:{end}  size={len(part)}')

            tasks = [
                infer_one_row(client, sem, row[ID_COL], row[SH_COL], str(row[NOTE_COL]))
                for _, row in part.iterrows()
            ]
            for coro in asyncio.as_completed(tasks):
                try:
                    rec = await coro
                except Exception as e:
                    rec = {ID_COL: None, 'error': repr(e), 'output_text': []}
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
    print(f'  Done with {site_key}')



async def main_async(site_keys: list[str]):
    ensure_dirs()
    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(STAGE1['max_in_flight'])

    for site_key in site_keys:
        await run_site(client, sem, site_key)


def parse_args():
    p = argparse.ArgumentParser(description='Stage 1 zero-shot screening')
    p.add_argument(
        '--sites', nargs='+', default=None,
        help='Site keys from config.SITES to run. Default: all sites.',
    )
    return p.parse_args()


def main():
    args = parse_args()
    site_keys = args.sites if args.sites else list(SITES.keys())
    # Validate
    unknown = [k for k in site_keys if k not in SITES]
    if unknown:
        raise SystemExit(f'Unknown site keys: {unknown}. '
                         f'Available: {list(SITES.keys())}')
    asyncio.run(main_async(site_keys))


if __name__ == '__main__':
    main()
