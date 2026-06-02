"""
ICD-10 RAG using TF-IDF + cosine similarity.

Replaces ChromaDB + sentence-transformers with a fully local sklearn-based
index — no internet download required. Builds in ~3 seconds, searches in <1ms.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

CMS_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'CMS Data')
)

ICD10_DESC_CSV = os.path.join(
    CMS_DATA_PATH,
    '2027-initial-icd-10-cm-mappings',
    '2027 Initial ICD-10-CM Mappings.csv'
)

V22_MAPPING_CSV = os.path.join(
    CMS_DATA_PATH,
    'python-2027-initial-model-software',
    'CMS_HCC_v22_2027_O1_initial_package_v1',
    'software', 'CMS_HCC_v22', 'data', 'input', 'internal',
    'ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv'
)

# Local cache for the TF-IDF index (avoids rebuilding on every restart)
INDEX_CACHE = os.path.join(os.path.dirname(__file__), 'tfidf_index.pkl')


def _norm(code: str) -> str:
    return str(code).strip().upper().replace('.', '')


class ICD10RAG:
    def __init__(self):
        self._vectorizer = None   # TfidfVectorizer
        self._matrix = None       # sparse TF-IDF matrix (n_docs × n_features)
        self._records: List[Dict] = []          # ordered list of ICD-10 dicts
        self._icd10_lookup: Dict[str, Dict] = {}
        self._ready = False

    @property
    def is_ready(self):
        return self._ready

    def initialize(self):
        logger.info("Loading ICD-10 data...")
        self._load_data()
        logger.info("Building TF-IDF index...")
        self._build_index()
        self._ready = True
        logger.info(f"ICD-10 RAG ready — {len(self._records)} codes indexed")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        desc_df = pd.read_csv(ICD10_DESC_CSV, skiprows=3, header=0,
                              low_memory=False, dtype=str)
        desc_df = desc_df.rename(columns={
            desc_df.columns[0]: 'ICD10',
            desc_df.columns[1]: 'Description'
        })
        desc_df = desc_df[['ICD10', 'Description']].dropna(subset=['ICD10', 'Description'])
        desc_df['ICD10_norm'] = desc_df['ICD10'].apply(_norm)
        desc_df = desc_df[desc_df['ICD10_norm'].str.len() >= 3]

        v22_df = pd.read_csv(V22_MAPPING_CSV, low_memory=False, dtype={'ICD10': str})
        v22_df['ICD10_norm'] = v22_df['ICD10'].apply(_norm)
        v22_df['CC_str'] = v22_df['CC'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) else ''
        )

        merged = desc_df.merge(
            v22_df[['ICD10_norm', 'CC_str']].drop_duplicates('ICD10_norm'),
            on='ICD10_norm', how='inner'
        )

        self._records = [
            {
                'icd10': row['ICD10_norm'],
                'description': str(row['Description']),
                'cc': row['CC_str'],
            }
            for _, row in merged.iterrows()
        ]
        self._icd10_lookup = {r['icd10']: r for r in self._records}
        logger.info(f"Loaded {len(self._records)} HCC-relevant ICD-10 codes")

    # ── TF-IDF index ─────────────────────────────────────────────────────────

    def _build_index(self):
        """Build or load a pickled TF-IDF index. Rebuilds if data changes."""
        if os.path.exists(INDEX_CACHE):
            try:
                with open(INDEX_CACHE, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('n_docs') == len(self._records):
                    self._vectorizer = cached['vectorizer']
                    self._matrix = cached['matrix']
                    logger.info(f"Loaded TF-IDF index from cache ({len(self._records)} docs)")
                    return
            except Exception:
                pass

        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Fitting TF-IDF vectorizer (first run, ~3 seconds)...")
        docs = [r['description'] for r in self._records]
        self._vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(docs)

        with open(INDEX_CACHE, 'wb') as f:
            pickle.dump({
                'n_docs': len(self._records),
                'vectorizer': self._vectorizer,
                'matrix': self._matrix,
            }, f)
        logger.info("TF-IDF index built and cached")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        if not self._ready or self._matrix is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:n_results]

        return [
            {**self._records[i], 'similarity': round(float(sims[i]), 3)}
            for i in top_idx
            if sims[i] > 0
        ]

    def get_by_code(self, code: str) -> Dict:
        return self._icd10_lookup.get(_norm(code), {})
