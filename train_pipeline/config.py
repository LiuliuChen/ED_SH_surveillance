"""
Central configuration for the self-harm screening pipeline.

Users editing this file for a new dataset typically only need to change:
  - ROOT_DIR (top of the file)
  - The SITES dictionary (entry per evaluation cohort)
  - VLLM_API_BASE if the model server is not on localhost

Everything else (sampling, feature budgets, seeds) has sensible defaults
and only needs to be touched for ablation studies.
"""
from pathlib import Path


"""Paths"""
# Root of the project. All other paths are derived from this unless overridden.
ROOT_DIR   = Path('/path/to/your/project')         # <-- EDIT for your environment
DATA_DIR   = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'outputs'

# Subdirectories for each stage's outputs
STAGE1_DIR  = OUTPUT_DIR / 'stage1_screening'
STAGE2_DIR  = OUTPUT_DIR / 'stage2_cot_extraction'
STAGE3_DIR  = OUTPUT_DIR / 'stage3_classifier'
EVAL_DIR    = OUTPUT_DIR / 'evaluation'


"""Data columns used for inference"""
DATA_COL = {
    'ID': 'uid',
    'note': 'triage_note',
    'self_harm_label': 'SH'
}


"""Hospital sites"""
SITES = {
    'rmh_train': {
        'parquet':      DATA_DIR / 'ED' / 'train' / 'rmh_chunk3.parquet',
        'stage1_jsonl': STAGE1_DIR / 'rmh_chunk3.jsonl',
        'stage2_jsonl': STAGE2_DIR / 'rmh_chunk3.jsonl',
        'role':         'train',
    },
    'rmh_test': {
        'parquet':      DATA_DIR / 'ED' / 'test' / 'rmh_test.parquet',
        'stage1_jsonl': STAGE1_DIR / 'rmh_test.jsonl',
        'stage2_jsonl': STAGE2_DIR / 'rmh_test.jsonl',
        'role':         'test',
    },
    # Cross-hospital and prospective sites: add as needed.
    # 'lrh_test': {...},
    # 'wh_test':  {...},
    # 'rmh_prospective': {...},
    # 'lrh_prospective': {...},
}


"""vLLM server"""
# Where the OpenAI-compatible vLLM server is reachable. Override via the
# OPENAI_BASE_URL environment variable in a SLURM script if needed.
VLLM_API_BASE = 'http://localhost:8000/v1'
VLLM_API_KEY  = 'sk-ignored'      # vLLM does not check; kept for client compat
MODEL_NAME    = 'gpt-oss-20b'     # served-model-name



"""Stage 1: zero-shot screening"""
STAGE1 = {
    'max_output_tokens': 512,
    'temperature':       0.3,
    'top_p':             0.95,
    'n_runs':            3,      # self-consistency samples per note
    'max_in_flight':     64,     # async concurrency (must be <= vLLM max-num-seqs)
    'chunk_size':        1000,   # checkpoint write frequency
    'reasoning_effort':  'low',  # 'low' / 'medium' / 'high'; reasoning models only
}


"""Stage 2: CoT structured extraction"""
STAGE2 = {
    'max_output_tokens': 1000,
    'temperature':       0.6,
    'top_p':             1.0,
    'n_runs':            5,
    'max_in_flight':     64,

    # Label sets for each SH indicator. Edit prompts.py if you change these.
    'method_labels': [
        'Asphyxia/asphyxiation', 'Battery with a blunt object',
        'Self-burning/self-immolation', 'Overdosing/self-poisoning',
        'Self-cutting with a sharp object', 'Drowning', 'Self-gassing',
        'Hanging', 'Insertion/ingestion of foreign bodies',
        'Jumping from heights', 'Jumping in front of vehicles',
        'Piercing with sharp implements', 'Strangulation',
        'Use of explosives/firearms', 'Vehicular impact',
        'Other method', 'Method not reported',
    ],
    'timing_labels':         ['Within 72h', 'Beyond 72h',
                              'Planning only', 'Not mentioned'],
    'intentionality_labels': ['Intentional', 'Accidental', 'Not mentioned'],
    'final_labels':          ['Self-harm: Yes', 'Self-harm: No',
                              'Self-harm: Ambiguous'],
}



"""Stage 3: classifier"""
STAGE3 = {
    # Classifier head. Currently supports 'lr_l2'. See stage3_classifier.py.
    'classifier': 'lr_l2',

    # Feature blocks to concatenate. Must match keys built in feature_extraction.py.
    'feature_keys': [
        'tfidf_note',
        'emb_note',
        'step3_tfidf_ev',
        'step3_tfidf_re',
        'summary_tfidf_ev',
        'summary_tfidf_re',
    ],

    # Feature budgets
    'embedding_model':    'all-MiniLM-L6-v2',
    'max_tfidf_note':     5000,
    'max_tfidf_ev':       1000,
    'max_tfidf_re':       1000,
    'max_tfidf_per_step': 500,
    'tfidf_ngram_note':   (1, 1),
    'tfidf_ngram_step':   (1, 2),

    # Training
    'class_weight':       'balanced',
    'max_iter':           1000,

    # Threshold selection on a stratified 20% holdout of training data
    'threshold_holdout_size': 0.2,

    # Where to save the trained pipeline (TF-IDF vectorisers + LR + threshold)
    'save_path': STAGE3_DIR / 'pipeline.joblib',
}



"""Evaluation"""
EVAL = {
    'n_bootstrap': 1000,
    'random_state': 42,
}


# =============================================================================
# Misc
# =============================================================================
RANDOM_STATE = 42

# CoT step names referenced by Stage 2 outputs and Stage 3 features
STEPS = ['step0', 'step1', 'step2', 'step3', 'step4', 'step5', 'summary']


# =============================================================================
# Convenience helpers
# =============================================================================
def ensure_dirs():
    """Create output directories if they do not exist. Call from entry points."""
    for d in (STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def train_sites():
    """Site keys whose role is 'train'."""
    return [k for k, v in SITES.items() if v['role'] == 'train']


def test_sites():
    """Site keys whose role is 'test'."""
    return [k for k, v in SITES.items() if v['role'] == 'test']
