"""
Stage 3: classifier training.
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SITES, STAGE3, RANDOM_STATE, train_sites, ensure_dirs, SH_COL
from feature_extraction import (
    load_site, fit_vectorisers, build_blocks, stack_features, load_encoder,
)


def make_classifier():
    name = STAGE3['classifier']
    if name == 'lr_l2':
        return LogisticRegression(
            penalty='l2',
            class_weight=STAGE3['class_weight'],
            max_iter=STAGE3['max_iter'],
            random_state=RANDOM_STATE,
        )
    raise ValueError(f'Unsupported classifier in STAGE3: {name}')


def find_threshold(X_train, y_train):
    """Stratified 80/20 holdout; fit on 80%, F1-optimal threshold on 20%."""
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=STAGE3['threshold_holdout_size'],
        random_state=RANDOM_STATE,
    )
    tr_idx, val_idx = next(sss.split(X_train, y_train))

    clf = make_classifier()
    clf.fit(X_train[tr_idx], y_train[tr_idx])
    proba_val = clf.predict_proba(X_train[val_idx])[:, 1]
    prec, rec, thresh = precision_recall_curve(y_train[val_idx], proba_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best = float(thresh[np.argmax(f1s[:-1])]) if len(thresh) > 0 else 0.5
    return best


def main():
    ensure_dirs()

    site_keys = train_sites()
    if not site_keys:
        raise SystemExit("No sites with role='train' in config.SITES. "
                         "Set at least one site to role='train'.")
    print(f'Training sites: {site_keys}')

    print('Loading MiniLM encoder...')
    encoder = load_encoder()

    print('Loading training data...')
    feat_dfs = []
    for site_key in site_keys:
        site = SITES[site_key]
        feat_df, _, _ = load_site(
            site['parquet'], site['stage1_jsonl'], site['stage2_jsonl'],
            name=site_key)
        feat_dfs.append(feat_df)
    train_df = (feat_dfs[0] if len(feat_dfs) == 1
                else __import__('pandas').concat(feat_dfs, ignore_index=True))
    print(f'Total training rows: {len(train_df):,}')

    print('Fitting vectorisers and building feature blocks...')
    vectorisers = fit_vectorisers(train_df)
    blocks      = build_blocks(train_df, encoder, vectorisers)
    X_train     = stack_features(blocks, STAGE3['feature_keys'])
    y_train     = train_df[SH_COL].astype(int).values

    print('Tuning threshold on 20% holdout...')
    threshold = find_threshold(X_train, y_train)
    print(f'  Selected threshold: {threshold:.3f}')

    print('Refitting classifier on full training set...')
    clf = make_classifier()
    clf.fit(X_train, y_train)

    bundle = {
        'classifier':   clf,
        'vectorisers':  vectorisers,
        'threshold':    threshold,
        'feature_keys': list(STAGE3['feature_keys']),
        'meta': {
            'training_sites': site_keys,
            'n_train':        int(len(train_df)),
            'n_features':     int(X_train.shape[1]),
        },
    }
    save_path: Path = STAGE3['save_path']
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f'Saved trained pipeline to {save_path}')


if __name__ == '__main__':
    main()
