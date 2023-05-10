from typing import Dict

import pandas as pd
from .clean_spoken import clean_spoken
from .restore_punctuation import restore_punc


def preprocessing(df: pd.DataFrame,
                  dialogue_column: str,
                  **kwargs) -> Dict:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    df_cleaned = clean_spoken(df, dialogue_column=dialogue_column, **kwargs)
    # df_punc = restore_punc(df_cleaned, dialogue_column='clean_dialogue', **kwargs)
    return {
        "df": df_cleaned
    }
