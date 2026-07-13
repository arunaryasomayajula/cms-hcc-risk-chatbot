import re
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# Reference data ships bundled under the repo's data/ folder so the project is
# self-contained. Override with the CMS_DATA_PATH env var to point at an external
# copy that uses the same layout (icd10_mappings/ and v22_internal/).
CMS_DATA_PATH = os.path.abspath(
    os.environ.get('CMS_DATA_PATH')
    or os.path.join(os.path.dirname(__file__), '..', 'data')
)
INTERNAL_DATA_PATH = os.path.join(CMS_DATA_PATH, 'v22_internal')

CUTOFF_DATE = datetime(2027, 2, 1)

CE_MODEL_COLS = [
    'COMMUNITY_NA', 'COMMUNITY_PBA', 'COMMUNITY_FBA',
    'COMMUNITY_ND', 'COMMUNITY_PBD', 'COMMUNITY_FBD', 'INSTITUTIONAL'
]
NE_MODEL_COLS = ['NEW_ENROLLEE', 'SNP_NEW_ENROLLEE']


def _calculate_age(dob: date, cutoff: datetime) -> int:
    years = cutoff.year - dob.year
    if cutoff.month < dob.month or (cutoff.month == dob.month and cutoff.day < dob.day):
        years -= 1
    return years


def _evaluate_age_rule(expr: str, age: int) -> bool:
    """Evaluate age condition expressions like 'age >= 18', 'age 50+', etc."""
    try:
        expr = str(expr).strip().lower()
        expr = re.sub(r'age\s*(\d+)\s*\+', r'age >= \1', expr)
        expr = re.sub(r'(?<![<>!])=', '==', expr)
        expr = re.sub(r'([<>=!]=?)', r' \1 ', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
        return bool(eval(expr, {"__builtins__": {}}, {"age": age}))
    except Exception:
        return True


def _cc_to_col(cc) -> str:
    """Convert CC value (possibly float) to clean column suffix string."""
    cc = str(cc).replace('_', '.')
    try:
        f = float(cc)
        if f.is_integer():
            return str(int(f))
        return ('%f' % f).rstrip('0').rstrip('.').replace('.', '_')
    except (ValueError, TypeError):
        return str(cc)


class HCCCalculator:
    def __init__(self):
        self.mappings_df: pd.DataFrame = None
        self.hierarchies_df: pd.DataFrame = None
        self.ce_factors_df: pd.DataFrame = None
        self.ne_factors_df: pd.DataFrame = None
        self.diag_categories_df: pd.DataFrame = None
        self.interactions_df: pd.DataFrame = None
        self._ready = False

    @property
    def is_ready(self):
        return self._ready

    def load_data(self):
        self.mappings_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv'),
            low_memory=False, dtype={'ICD10': str}
        )
        self.mappings_df['ICD10'] = self.mappings_df['ICD10'].str.strip().str.upper()

        self.hierarchies_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'V22_HCC_Hierarchies.csv')
        )

        self.ce_factors_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'V22_CE_Relative_Factors.csv')
        ).dropna(subset=['Variable'])
        self.ce_factors_df['Variable'] = self.ce_factors_df['Variable'].str.strip()

        self.ne_factors_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'V22_NE_Relative_Factors.csv')
        ).dropna(subset=['Variable'])
        self.ne_factors_df['Variable'] = self.ne_factors_df['Variable'].str.strip()

        self.diag_categories_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'V22_Diagnosis_Categories.csv')
        )

        self.interactions_df = pd.read_csv(
            os.path.join(INTERNAL_DATA_PATH, 'V22_Interactions.csv')
        ).dropna(subset=['interaction'])

        self._ready = True
        logger.info("HCC Calculator loaded all reference data")

    def _get_age_sex_vars(self, age: int, sex: int) -> Dict[str, int]:
        age_ranges = [
            (0, 34), (35, 44), (45, 54), (55, 59), (60, 64),
            (65, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 94), (95, None)
        ]
        sex_char = 'M' if sex == 1 else 'F'
        vars_dict = {}
        for s, e in age_ranges:
            for sc in ['M', 'F']:
                vars_dict[f'{sc}{s}_{e if e is not None else "GT"}'] = 0
        for s, e in age_ranges:
            if age >= s and (e is None or age <= e):
                vars_dict[f'{sex_char}{s}_{e if e is not None else "GT"}'] = 1
                break
        return vars_dict

    def _get_ne_age_sex_vars(self, age: int, sex: int, orec: int) -> Dict[str, int]:
        ne_ranges = [
            (0, 34), (35, 44), (45, 54), (55, 59), (60, 64),
            (65, 65), (66, 66), (67, 67), (68, 68), (69, 69),
            (70, 74), (75, 79), (80, 84), (85, 89), (90, 94), (95, None)
        ]
        sex_char = 'M' if sex == 1 else 'F'
        vars_dict = {}
        for s, e in ne_ranges:
            for sc in ['M', 'F']:
                suffix = f'{s}' if s == e else f'{s}_{e if e is not None else "GT"}'
                vars_dict[f'NE{sc}{suffix}'] = 0

        label = None
        for s, e in ne_ranges:
            suffix = f'{s}' if s == e else f'{s}_{e if e is not None else "GT"}'
            if age == 64:
                if s == 60 and e == 64 and orec != 0:
                    label = f'NE{sex_char}60_64'
                    break
                elif s == 65 and e == 65 and orec == 0:
                    label = f'NE{sex_char}65'
                    break
            elif age >= s and (e is None or age <= e):
                label = f'NE{sex_char}{suffix}'
                break

        if label and label in vars_dict:
            vars_dict[label] = 1
        return vars_dict

    def _map_icd10_to_ccs(self, icd10_codes: List[str], age: int, sex: int,
                           switch_edits: bool = True) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
        all_ccs = {f'CC{_cc_to_col(row["HCC"].replace("HCC", ""))}': 0
                   for _, row in self.hierarchies_df.iterrows()}
        icd10_map: Dict[str, List[int]] = {}

        for raw_code in icd10_codes:
            code = str(raw_code).strip().upper().replace('.', '')
            rows = self.mappings_df[self.mappings_df['ICD10'] == code]
            for _, mrow in rows.iterrows():
                try:
                    cc_num = int(float(mrow['CC']))
                except (ValueError, TypeError):
                    continue
                cc_col = f'CC{cc_num}'

                if switch_edits:
                    mce = mrow.get('MCE_AGE_CONDITION')
                    if pd.notna(mce) and not _evaluate_age_rule(str(mce), age):
                        continue

                age_cond = mrow.get('AGE_EDIT_CONDITION')
                if pd.notna(age_cond) and not _evaluate_age_rule(str(age_cond), age):
                    continue

                sex_cond = mrow.get('SEX_EDIT_CONDITION')
                if pd.notna(sex_cond):
                    try:
                        if int(float(sex_cond)) != sex:
                            continue
                    except (ValueError, TypeError):
                        pass

                if cc_col in all_ccs:
                    all_ccs[cc_col] = 1
                icd10_map.setdefault(code, []).append(cc_num)

        return all_ccs, icd10_map

    def _apply_hierarchies(self, cc_flags: Dict[str, int]) -> Dict[str, int]:
        """CC columns → HCC columns with hierarchies applied."""
        # First rename CC → HCC
        hcc_flags = {k.replace('CC', 'HCC', 1): v for k, v in cc_flags.items()}
        hcc_col_prefix = 'HCC'
        excl_cols = self.hierarchies_df.columns.drop(hcc_col_prefix).tolist()

        hcc_init = dict(hcc_flags)
        hcc_final = dict(hcc_flags)

        for _, row in self.hierarchies_df.iterrows():
            raw_val = str(row[hcc_col_prefix]).replace(hcc_col_prefix, '')
            primary_col = f'{hcc_col_prefix}{_cc_to_col(raw_val)}'
            if hcc_init.get(primary_col, 0) != 1:
                continue
            for excl_col in excl_cols:
                excl_val = row[excl_col]
                if pd.isna(excl_val):
                    continue
                excl_str = str(excl_val).replace(hcc_col_prefix, '')
                excl_hcc_col = f'{hcc_col_prefix}{_cc_to_col(excl_str)}'
                if excl_hcc_col in hcc_final and hcc_init.get(excl_hcc_col, 0) == 1:
                    hcc_final[excl_hcc_col] = 0

        return hcc_final

    def _get_diag_categories(self, hcc_flags: Dict[str, int]) -> Dict[str, int]:
        cat_flags = {}
        hcc_cols_in_df = self.diag_categories_df.columns.drop('diag_category').tolist()
        for _, row in self.diag_categories_df.iterrows():
            cat = str(row['diag_category']).strip()
            hccs = [str(row[c]).strip() for c in hcc_cols_in_df if pd.notna(row.get(c))]
            cat_flags[cat] = 1 if any(hcc_flags.get(h, 0) == 1 for h in hccs) else 0
        return cat_flags

    def _get_interactions(self, all_flags: Dict[str, int]) -> Dict[str, int]:
        result = {}
        for _, row in self.interactions_df.iterrows():
            var = str(row['interaction']).strip()
            v1 = str(row['var_1']).strip()
            v2 = str(row['var_2']).strip()
            result[var] = int(all_flags.get(v1, 0) * all_flags.get(v2, 0))
        return result

    def _score(self, features: Dict, factors_df: pd.DataFrame, model_col: str) -> float:
        total = 0.0
        for _, row in factors_df.iterrows():
            var = str(row['Variable']).strip()
            coef = row.get(model_col)
            if pd.notna(coef):
                total += float(features.get(var, 0)) * float(coef)
        return round(total, 3)

    def _applicable_model(self, age: int, ltimcaid: int, dual_status: int) -> str:
        if ltimcaid == 1:
            return 'INSTITUTIONAL'
        aged = age >= 65
        if aged:
            return {2: 'COMMUNITY_FBA', 1: 'COMMUNITY_PBA'}.get(dual_status, 'COMMUNITY_NA')
        return {2: 'COMMUNITY_FBD', 1: 'COMMUNITY_PBD'}.get(dual_status, 'COMMUNITY_ND')

    def calculate(self, demographics: dict, icd10_codes: List[str]) -> dict:
        if not self._ready:
            raise RuntimeError("HCCCalculator not loaded")

        dob_str = demographics.get('dob', '1950-01-01')
        try:
            dob = datetime.strptime(str(dob_str), '%Y-%m-%d').date()
        except ValueError:
            dob = date(1950, 1, 1)

        age = _calculate_age(dob, CUTOFF_DATE)
        sex = int(demographics.get('sex', 2))
        orec = int(demographics.get('orec', 0))
        ltimcaid = int(demographics.get('ltimcaid', 0))
        nemcaid = int(demographics.get('nemcaid', 0))
        dual_status = int(demographics.get('dual_status', 0))

        disabl = 1 if (age < 65 and orec in [1, 2, 3]) else 0
        origdis = 1 if (orec == 1 and disabl == 0) else 0
        ne_origdis = 1 if (age >= 65 and orec == 1) else 0

        age_sex_vars = self._get_age_sex_vars(age, sex)

        cc_flags, icd10_to_cc_map = self._map_icd10_to_ccs(icd10_codes, age, sex)
        hcc_flags = self._apply_hierarchies(cc_flags)
        diag_cat_flags = self._get_diag_categories(hcc_flags)

        all_features: Dict[str, int] = {
            **age_sex_vars,
            'DISABL': disabl,
            'ORIGDIS': origdis,
            'ORIGDS': origdis,
            'OriginallyDisabled_Female': 1 if (sex == 2 and origdis == 1) else 0,
            'OriginallyDisabled_Male': 1 if (sex == 1 and origdis == 1) else 0,
            'LTIMCAID': ltimcaid,
            **hcc_flags,
            **diag_cat_flags,
        }
        all_features.update(self._get_interactions(all_features))

        ce_scores = {col: self._score(all_features, self.ce_factors_df, col) for col in CE_MODEL_COLS}

        # New enrollee features
        ne_age_sex = self._get_ne_age_sex_vars(age, sex, orec)
        ne_mcaid = {
            'NMCAID_NORIGDIS': 1 if (nemcaid == 0 and ne_origdis == 0) else 0,
            'MCAID_NORIGDIS':  1 if (nemcaid == 1 and ne_origdis == 0) else 0,
            'NMCAID_ORIGDIS':  1 if (nemcaid == 0 and ne_origdis == 1 and age >= 65) else 0,
            'MCAID_ORIGDIS':   1 if (nemcaid == 1 and ne_origdis == 1 and age >= 65) else 0,
        }
        ne_features: Dict[str, int] = {**ne_age_sex, **ne_mcaid}
        for age_var, age_val in ne_age_sex.items():
            for int_var, int_val in ne_mcaid.items():
                if int_var in ('NMCAID_ORIGDIS', 'MCAID_ORIGDIS'):
                    suffix = age_var.split('_')[-1]
                    try:
                        if suffix != 'GT' and int(suffix) < 65:
                            continue
                    except ValueError:
                        pass
                ne_features[f'{int_var}_{age_var}'] = age_val * int_val

        ne_scores = {col: self._score(ne_features, self.ne_factors_df, col) for col in NE_MODEL_COLS}

        applicable = self._applicable_model(age, ltimcaid, dual_status)
        active_hccs = {hcc: 1 for hcc, v in hcc_flags.items() if v == 1}

        # Build HCC label map from CE factors for display
        hcc_labels = {}
        for _, row in self.ce_factors_df.iterrows():
            var = str(row['Variable']).strip()
            if var.startswith('HCC') and pd.notna(row.get('Label')):
                hcc_labels[var] = str(row['Label']).strip().strip('"')

        active_hcc_details = [
            {'hcc': hcc, 'description': hcc_labels.get(hcc, '')}
            for hcc in sorted(active_hccs.keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0))
        ]

        return {
            'demographics_derived': {
                'age': age, 'sex_label': 'Male' if sex == 1 else 'Female',
                'orec_label': {0: 'Aged/OASI', 1: 'Disabled/DIB', 2: 'ESRD', 3: 'DIB+ESRD'}.get(orec, 'Unknown'),
                'disabl': disabl, 'origdis': origdis,
            },
            'icd10_to_cc': icd10_to_cc_map,
            'active_hccs': active_hcc_details,
            'diag_categories_triggered': [k for k, v in diag_cat_flags.items() if v == 1],
            'interactions_triggered': [k for k, v in self._get_interactions(all_features).items() if v == 1],
            'ce_scores': ce_scores,
            'ne_scores': ne_scores,
            'all_scores': {**ce_scores, **ne_scores},
            'applicable_model': applicable,
            'applicable_score': ce_scores.get(applicable, ne_scores.get(applicable, 0.0)),
        }
