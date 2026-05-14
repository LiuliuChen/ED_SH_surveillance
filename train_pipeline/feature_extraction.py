"""
Feature extraction for Stage 3.

Reads a parquet (gold labels + triage notes) and the matching Stage 2 jsonl
(CoT outputs), produces a feature dataframe suitable for the classifier:

  - Raw note TF-IDF
  - MiniLM sentence embedding of the note
  - Per-step TF-IDF over evidence (ev) and reasoning (re) text from the CoT output
  - Step-label encodings (intent, timing, method, etc.)

Used by both stage3_classifier.py (training) and the evaluation scripts.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STAGE3, STEPS


NOTE_COL = 'triage_note'


# =============================================================================
# Stage 1 + Stage 2 -> per-uid feature dictionary
# =============================================================================
def extract_cot_features(stage2_jsonl: Path) -> pd.DataFrame:
    """Parse stage 2 jsonl into a DataFrame, one row per uid.

    For each uid, computes:
      - majority labels for each step (encoded as integers where appropriate)
      - flattened evidence + reasoning text (pooled across runs)
      - per-step evidence + reasoning text (pooled across runs, deduplicated)
      - aggregate confidence (fraction of yes votes for final_decision)
    """
    rows = []
    with open(stage2_jsonl) as f:
        for line in f:
            record = json.loads(line)
            uid = record['uid']
            steps = record.get('steps', {})

            def get_label(step):
                return steps.get(step, {}).get('label', '')

            fd_votes = steps.get('final_decision', {}).get('votes', {})
            n_parsed = record.get('n_parsed', 0)
            n_yes    = fd_votes.get('Self-harm: Yes', 0)
            n_ambig  = fd_votes.get('Self-harm: Ambiguous', 0)

            all_ev, all_re = [], []
            step_ev = defaultdict(list)
            step_re = defaultdict(list)

            for resp_str in record.get('responses', []):
                try:
                    resp = json.loads(resp_str) if isinstance(resp_str, str) else resp_str
                except Exception:
                    continue
                if not isinstance(resp, dict):
                    continue
                for step in STEPS:
                    block = resp.get(step, {})
                    if not isinstance(block, dict):
                        continue
                    ev = str(block.get('evidence', '') or '').strip()
                    if ev and ev not in ('None', 'null', ''):
                        all_ev.append(ev)
                        if ev not in step_ev[step]:
                            step_ev[step].append(ev)
                    re_text = str(block.get('reasoning', '') or '').strip()
                    if re_text and re_text not in ('None', 'null', ''):
                        all_re.append(re_text)
                        if re_text not in step_re[step]:
                            step_re[step].append(re_text)

            row = {
                'uid':             uid,
                'kw_cues_bin':     int(get_label('step0') == 'Yes'),
                'act_on_body_bin': int(get_label('step1') == 'Yes'),
                'injury_bin':      int(get_label('step2') == 'Yes'),
                'intent_enc':      {'Intentional': 2, 'Accidental': 1}.get(get_label('step5'), 0),
                'timing_enc':      {'Within 72h': 3, 'Beyond 72h': 2, 'Planning only': 1}.get(get_label('step4'), 0),
                'method_enc':      0 if get_label('step3') in ('Method not reported', '') else 1,
                'llm_pred':        int(n_yes > n_parsed / 2),
                'llm_ambig':       int(n_ambig > 0),
                'confidence':      n_yes / n_parsed if n_parsed > 0 else 0.0,
                'n_parsed':        n_parsed,
                'all_evidence':    ' '.join(dict.fromkeys(all_ev)),
                'all_reasoning':   ' '.join(dict.fromkeys(all_re)),
            }
            for step in STEPS:
                row[f'{step}_ev_text'] = ' '.join(step_ev[step])
                row[f'{step}_re_text'] = ' '.join(step_re[step])
            rows.append(row)
    return pd.DataFrame(rows)


def load_stage1_kept_uids(stage1_jsonl: Path) -> set:
    """Return uids whose Stage 1 majority vote was self-harm-related (not 'no')."""
    from collections import Counter
    from utils import safe_parse

    keep = set()
    with open(stage1_jsonl) as f:
        for line in f:
            try:
                data = json.loads(line)
                parsed = [safe_parse(t) for t in data.get('output_text', [])]
                parsed = [p for p in parsed if p and 'self-harm-related' in p]
                if not parsed:
                    continue
                votes = Counter([p['self-harm-related'] for p in parsed])
                vote = votes.most_common(1)[0][0]
                if vote.lower() != 'no':
                    keep.add(data['uid'])
            except Exception:
                pass
    return keep


def load_site(parquet_path: Path, stage1_jsonl: Path, stage2_jsonl: Path, name='site'):
    """Load one site for Stage 3.

    Returns:
      feat_df       : DataFrame restricted to Stage-1-kept uids, with CoT features merged
      full_labels   : SH labels for the full parquet (used for end-to-end scoring)
      filt_indices  : positions of feat_df rows inside the full parquet
    """
    full_df = pd.read_parquet(parquet_path)
    full_df[NOTE_COL] = full_df[NOTE_COL].fillna('')
    full_df = full_df[full_df[NOTE_COL].str.strip() != ''].reset_index(drop=True)
    print(f'  [{name}] {len(full_df):,} notes, '
          f'{int(full_df["SH"].sum()):,} positive '
          f'({full_df["SH"].mean():.3%})')

    stage1_uids = load_stage1_kept_uids(stage1_jsonl)
    cot_df = extract_cot_features(stage2_jsonl)

    feat_df = full_df[full_df['uid'].isin(stage1_uids)].copy().reset_index(drop=True)
    feat_df = feat_df.merge(cot_df, on='uid', how='inner')
    print(f'  [{name}] Stage 1 + Stage 2 retained: {len(feat_df):,} notes, '
          f'{int(feat_df["SH"].sum()):,} positive '
          f'({feat_df["SH"].mean():.3%})')

    full_labels = full_df['SH'].astype(int).values
    uid_to_idx  = {uid: i for i, uid in enumerate(full_df['uid'])}
    filt_indices = np.array([uid_to_idx[uid] for uid in feat_df['uid']])
    return feat_df, full_labels, filt_indices


# =============================================================================
# Feature blocks
# =============================================================================
def fit_vectorisers(feat_df: pd.DataFrame):
    """Fit all TF-IDF vectorisers on a training feat_df."""
    v = {}
    v['tfidf_note'] = TfidfVectorizer(
        max_features=STAGE3['max_tfidf_note'],
        ngram_range=STAGE3['tfidf_ngram_note'],
        sublinear_tf=True,
    ).fit(feat_df[NOTE_COL].fillna('').tolist())

    v['tfidf_ev'] = TfidfVectorizer(
        max_features=STAGE3['max_tfidf_ev'], ngram_range=(1, 1), sublinear_tf=True,
    ).fit(feat_df['all_evidence'].fillna('').tolist())

    v['tfidf_re'] = TfidfVectorizer(
        max_features=STAGE3['max_tfidf_re'], ngram_range=(1, 1), sublinear_tf=True,
    ).fit(feat_df['all_reasoning'].fillna('').tolist())

    for step in STEPS:
        v[f'{step}_tfidf_ev'] = TfidfVectorizer(
            max_features=STAGE3['max_tfidf_per_step'],
            ngram_range=STAGE3['tfidf_ngram_step'],
            sublinear_tf=True, min_df=2,
        ).fit(feat_df[f'{step}_ev_text'].fillna(''))
        v[f'{step}_tfidf_re'] = TfidfVectorizer(
            max_features=STAGE3['max_tfidf_per_step'],
            ngram_range=STAGE3['tfidf_ngram_step'],
            sublinear_tf=True, min_df=2,
        ).fit(feat_df[f'{step}_re_text'].fillna(''))
    return v


def build_blocks(feat_df: pd.DataFrame, encoder: SentenceTransformer, vectorisers: dict):
    """Transform feat_df into a dict of named sparse feature blocks."""
    notes  = feat_df[NOTE_COL].fillna('').tolist()
    all_ev = feat_df['all_evidence'].fillna('').tolist()
    all_re = feat_df['all_reasoning'].fillna('').tolist()

    blocks = {}
    blocks['tfidf_note'] = sp.csr_matrix(vectorisers['tfidf_note'].transform(notes))
    blocks['tfidf_ev']   = sp.csr_matrix(vectorisers['tfidf_ev'].transform(all_ev))
    blocks['tfidf_re']   = sp.csr_matrix(vectorisers['tfidf_re'].transform(all_re))
    for step in STEPS:
        blocks[f'{step}_tfidf_ev'] = sp.csr_matrix(
            vectorisers[f'{step}_tfidf_ev'].transform(
                feat_df[f'{step}_ev_text'].fillna('')))
        blocks[f'{step}_tfidf_re'] = sp.csr_matrix(
            vectorisers[f'{step}_tfidf_re'].transform(
                feat_df[f'{step}_re_text'].fillna('')))

    blocks['emb_note'] = sp.csr_matrix(encoder.encode(
        notes, batch_size=128, show_progress_bar=False, convert_to_numpy=True))
    return blocks


def stack_features(blocks: dict, keys: list[str]):
    """Horizontally stack the requested feature blocks into one matrix."""
    mats = [blocks[k] for k in keys if k in blocks]
    return mats[0] if len(mats) == 1 else hstack(mats)


def load_encoder() -> SentenceTransformer:
    return SentenceTransformer(STAGE3['embedding_model'])
