import re
from typing import *

import pandas as pd


def term_replace(text: str, contractions: Dict) -> str:
    for k, v in contractions.items():
        if re.search(k, text.lower()):
            text = re.sub(k, v, text)
    return text


def extra_rule(ip_string:str):
    ip_string = ip_string.split("  ")[0]
    ip_string = re.sub(r"(\d+) and half year old", r"\1.5 year-old", ip_string)
    ip_string = re.sub(r"(\d+) out of (\d+)", r"\1/\2", ip_string)
    ip_string = re.sub(r"(\d) thousand(?: (?:and )?(\d))?", lambda m: m.group(1) + "00" + (m.group(2) or "0"),
                       ip_string)
    ip_string = re.sub(r"(\d+)-(\d+)", lambda m: m.group(1)[0] + m.group(2), ip_string)
    ip_string = re.sub(r'you have', 'the patient has', ip_string)
    ip_string = re.sub(r'you( are|\'re)', 'the patient is', ip_string)
    ip_string = re.sub(r'you do', 'the patient does', ip_string)
    ip_string = re.sub(r'(you|you guys|You)\b', 'the patient', ip_string)
    ip_string = re.sub(r'your', 'the patient\'s', ip_string)
    return ip_string

def for_run_3(ip_string):
    ip_string = re.sub(r"(\d+) and half year old", r"\1.5 year-old", ip_string)
    ip_string = re.sub(r"(\d+) out of (\d+)", r"\1/\2", ip_string)
    ip_string = re.sub(r"(\d) thousand(?: (?:and )?(\d))?", lambda m: m.group(1) + "00" + (m.group(2) or "0"), ip_string)
    ip_string = re.sub(r"(\d+)-(\d+)", lambda m: m.group(1)[0] + m.group(2), ip_string)
    ip_string = re.sub(r'you have', 'the patient has', ip_string)
    ip_string = re.sub(r'you( are|\'re)', 'the patient is', ip_string)
    ip_string = re.sub(r'you do', 'the patient does', ip_string)
    ip_string = re.sub(r'(you|you guys|You)\b', 'the patient', ip_string)
    ip_string = re.sub(r'your', 'the patient\'s', ip_string)
    ip_string = re.sub(r'\n{3,}', r'\n\n', ip_string)
    ip_string = re.sub(r'(\s) +', r'\1', ip_string)
    ip_string = re.sub(r' *([\.,])( ?[\.,])+', r'\1', ip_string)
    ip_string = re.sub(r"Â - \.", "", ip_string)
    return ip_string


def postprocessing(df: pd.DataFrame,
                   dialogue_column: str,
                   index_column: str,
                   FULL_CONTRACTIONS: Dict,
                   is_run_3:bool=False,
                   **kwargs) -> dict:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    assert index_column in df.columns, f" Column '{index_column}' not found in the given dataframe!"

    def _postprocess(row):
        text = row[dialogue_column]
        row[dialogue_column] = term_replace(text, FULL_CONTRACTIONS)
        if is_run_3:
            row[dialogue_column] = for_run_3(row[dialogue_column])
            return row
        row[dialogue_column] = extra_rule(row[dialogue_column])
        return row

    post_process_df = df.apply(_postprocess, axis=1)

    return {
        "df": post_process_df
    }
