import os
import logging
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

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')


def _norm(code: str) -> str:
    return str(code).strip().upper().replace('.', '')


class ICD10RAG:
    def __init__(self):
        self._client = None
        self._collection = None
        self._icd10_lookup: Dict[str, Dict] = {}
        self._ready = False

    @property
    def is_ready(self):
        return self._ready

    def initialize(self):
        logger.info("Loading ICD-10 data...")
        self._load_data()
        logger.info("Building/loading ChromaDB vector index...")
        self._setup_chroma()
        self._ready = True
        logger.info(f"ICD-10 RAG ready — {len(self._icd10_lookup)} codes indexed")

    def _load_data(self):
        # Load description CSV (3 comment rows before actual header)
        desc_df = pd.read_csv(ICD10_DESC_CSV, skiprows=3, header=0,
                              low_memory=False, dtype=str)
        # First col = code, second = description (column names may contain newlines)
        desc_df = desc_df.rename(columns={
            desc_df.columns[0]: 'ICD10',
            desc_df.columns[1]: 'Description'
        })
        desc_df = desc_df[['ICD10', 'Description']].dropna(subset=['ICD10', 'Description'])
        desc_df['ICD10_norm'] = desc_df['ICD10'].apply(_norm)
        desc_df = desc_df[desc_df['ICD10_norm'].str.len() >= 3]

        # Load V22 CC mapping (only HCC-relevant codes)
        v22_df = pd.read_csv(V22_MAPPING_CSV, low_memory=False, dtype={'ICD10': str})
        v22_df['ICD10_norm'] = v22_df['ICD10'].apply(_norm)
        v22_df['CC_str'] = v22_df['CC'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) else ''
        )

        # Keep only codes that have an HCC mapping
        merged = desc_df.merge(
            v22_df[['ICD10_norm', 'CC_str']].drop_duplicates('ICD10_norm'),
            on='ICD10_norm', how='inner'
        )

        self._icd10_lookup = {
            row['ICD10_norm']: {
                'icd10': row['ICD10_norm'],
                'description': str(row['Description']),
                'cc': row['CC_str'],
            }
            for _, row in merged.iterrows()
        }
        logger.info(f"Loaded {len(self._icd10_lookup)} HCC-relevant ICD-10 codes")

    def _setup_chroma(self):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        ef = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        name = 'icd10_v22_2027_v1'

        # Try loading existing collection
        try:
            col = self._client.get_collection(name=name, embedding_function=ef)
            if col.count() >= len(self._icd10_lookup) * 0.9:
                self._collection = col
                logger.info(f"Loaded existing index ({col.count()} documents)")
                return
            self._client.delete_collection(name)
        except Exception:
            pass

        # Build new collection
        logger.info("Building new ChromaDB index — this takes ~2-5 min on first run...")
        self._collection = self._client.create_collection(name=name, embedding_function=ef)

        items = list(self._icd10_lookup.values())
        batch = 500
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            self._collection.add(
                ids=[it['icd10'] for it in chunk],
                documents=[it['description'] for it in chunk],
                metadatas=chunk,
            )
            if i % 5000 == 0 and i > 0:
                logger.info(f"  Indexed {i}/{len(items)}...")
        logger.info("ChromaDB index built")

    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        if not self._ready or not self._collection:
            return []
        n = min(n_results, self._collection.count())
        results = self._collection.query(query_texts=[query], n_results=n)
        hits = []
        for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
            hits.append({
                'icd10': meta.get('icd10', ''),
                'description': meta.get('description', ''),
                'cc': meta.get('cc', ''),
                'similarity': round(max(0.0, 1.0 - dist), 3),
            })
        return hits

    def get_by_code(self, code: str) -> Dict:
        """Direct lookup by ICD-10 code."""
        return self._icd10_lookup.get(_norm(code), {})
