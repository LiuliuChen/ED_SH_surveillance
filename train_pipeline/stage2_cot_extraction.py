"""
Stage 2: CoT structured extraction.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    VLLM_API_BASE, VLLM_API_KEY, MODEL_NAME,
    STAGE2, SITES, STEPS, ensure_dirs,
    ID_COL, NOTE_COL, SH_COL,
)
from prompts import build_stage2_prompt
from utils import safe_parse


"""
Identify Stage 1 SH candidates from the screening jsonl
"""

def majority_vote(votes):
    return Counter(votes).most_common(1)[0][0]


def get_sh_candidate_uids(stage1_jsonl: Path) -> list[str]:
    """Return uids whose Stage 1 majority vote is not 'no'.

    The Stage 1 output_text field holds N_RUNS JSON strings; we parse each and
    take the majority of the `self-harm-related` field. Records marked 'yes'
    or 'unsure' are passed to Stage 2; 'no' is filtered out.
    """
    keep = []
    skipped = 0
    with open(stage1_jsonl) as f:
        for line in f:
            try:
                data = json.loads(line)
                parsed = [safe_parse(t) for t in data.get('output_text', [])]
                parsed = [p for p in parsed if p and 'self-harm-related' in p]
                if not parsed:
                    skipped += 1
                    continue
                vote = majority_vote([p['self-harm-related'] for p in parsed])
                if vote.lower() != 'no':
                    keep.append(data[ID_COL])
            except Exception:
                skipped += 1
    if skipped:
        print(f'  Stage 1 parse skipped: {skipped:,}')
    return keep



"""
# Majority vote across self-consistency runs (per-step)
"""

STEP_KEYS = STEPS + ['final_decision']


def majority_vote_stepwise(parsed_responses):
    """For each step in STEP_KEYS, take the majority label across n_runs."""
    out = {}
    for key in STEP_KEYS:
        label_counter = Counter()
        reasoning_by_label = {}
        for p in parsed_responses:
            node = p.get(key)
            if node is None:
                continue
            if key == 'final_decision':
                label, reasoning = (node if isinstance(node, str) else None), ''
            elif isinstance(node, dict):
                label = node.get('value') or node.get('label')
                reasoning = node.get('reasoning') or node.get('evidence') or ''
            else:
                continue
            if label:
                label_counter[label] += 1
                reasoning_by_label.setdefault(label, []).append(reasoning.strip())
        if not label_counter:
            continue
        majority_label, _ = label_counter.most_common(1)[0]
        best_reasoning = (
            Counter(reasoning_by_label[majority_label]).most_common(1)[0][0]
            if reasoning_by_label.get(majority_label) else ''
        )
        out[key] = {
            'label':     majority_label,
            'reasoning': best_reasoning,
            'votes':     dict(label_counter),
        }
    return out



"""Async inference"""

async def query_one_sample(client, sem, messages):
    """One chat-completion call. Returns response text or None on error."""
    async with sem:
        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=STAGE2['temperature'],
                max_tokens=STAGE2['max_output_tokens'],
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f'  [API Error] {repr(e)[:200]}')
            return None


async def process_row(client, sem, row):
    """Run N_RUNS self-consistency calls concurrently for one note."""
    uid      = row[ID_COL]
    note     = row[NOTE_COL]
    sh_label = row[SH_COL]

    messages = build_stage2_prompt(note)
    tasks = [query_one_sample(client, sem, messages) for _ in range(STAGE2['n_runs'])]
    responses = await asyncio.gather(*tasks)

    parsed = [safe_parse(r) for r in responses]
    parsed = [p for p in parsed if p is not None]
    majority = majority_vote_stepwise(parsed) if parsed else {}

    return {
        ID_COL: uid,
        SH_COL: int(sh_label) if sh_label is not None else None,
        'steps':     majority,
        'n_parsed':  len(parsed),
        'responses': responses,
    }



# Resume mode
def load_processed_uids(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    processed = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                processed.add(json.loads(line)[ID_COL])
            except Exception:
                pass
    return processed



# Per-site runner
async def run_site(client, sem, site_key: str):
    site         = SITES[site_key]
    parquet_path = site['parquet']
    stage1_path  = site['stage1_jsonl']
    out_path     = site['stage2_jsonl']

    print('=' * 70)
    print(f'Site: {site_key}')
    print(f'  parquet: {parquet_path}')
    print(f'  stage1:  {stage1_path}')
    print(f'  output:  {out_path}')
    print('=' * 70)

    if not stage1_path.exists():
        print(f'  ERROR: Stage 1 jsonl not found; run Stage 1 first.')
        return

    sh_uids = get_sh_candidate_uids(stage1_path)
    print(f'  Stage 1 SH candidates: {len(sh_uids):,}')
    if not sh_uids:
        print('  Nothing to do.')
        return

    df = pd.read_parquet(parquet_path, columns=[ID_COL, NOTE_COL, SH_COL])
    df = df[df[ID_COL].isin(sh_uids)].reset_index(drop=True)

    done = load_processed_uids(out_path)
    if done:
        before = len(df)
        df = df[~df[ID_COL].isin(done)].reset_index(drop=True)
        print(f'  resume: {len(done):,} uids already done; '
              f'{before:,} -> {len(df):,} remaining')

    if len(df) == 0:
        print('  Nothing to do.')
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'a', encoding='utf-8', buffering=1) as fout:
        records = df.to_dict('records')
        tasks = [process_row(client, sem, row) for row in records]
        done_count = 0
        for coro in asyncio.as_completed(tasks):
            try:
                rec = await coro
            except Exception as e:
                print(f'  [worker error] {repr(e)[:200]}')
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done_count += 1
            if done_count % 100 == 0:
                fout.flush()
                print(f'  {done_count}/{len(records)} done')
        fout.flush()
        print(f'  {done_count}/{len(records)} done. Finished.')



async def main_async(site_keys):
    ensure_dirs()
    client = AsyncOpenAI(base_url=VLLM_API_BASE, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(STAGE2['max_in_flight'])
    for site_key in site_keys:
        await run_site(client, sem, site_key)


def parse_args():
    p = argparse.ArgumentParser(description='Stage 2 CoT structured extraction')
    p.add_argument('--sites', nargs='+', default=None,
                   help='Site keys from config.SITES to run. Default: all.')
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
