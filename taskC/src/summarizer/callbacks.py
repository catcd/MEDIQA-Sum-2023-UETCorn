from typing import Any, Dict, List, Union

from .soft_cutoff import soft_cutoff
from .util import *
from ..util import split_sentence

_MEANINGLESS_TEXT = '<--NOISY-->'


def term_replace(to_return: str,
                 contractions: List[Dict], **callback_kwargs) -> str:
    text = to_return
    for contraction in contractions:
        for k, v in contraction.items():
            # if re.search(k, text.lower()):
            text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text


def do_nothing(to_return: Any = None, context: dict = None, **callback_kwargs) -> str:
    return to_return


def add_default_value(to_return: str, default: str, **callback_kwargs) -> str:
    if not to_return.strip():
        return default
    return to_return


def remove_noisy_sentence(context: dict = None,
                          use_soft_cutoff: bool = False,
                          drop_roles: Union[str, List[str]] = None,
                          spoken_terms: List[str] = None,
                          wanted_like_sentences: List[str] = None,
                          pos_thresh: float = 0.1,
                          unwanted_like_sentences: List[str] = None,
                          neg_thresh: float = 0.7,
                          model: str = 'paraphrase-MiniLM-L6-v2',
                          device: str = 'auto',
                          **callback_kwargs) -> str:
    text = context["raw_extract"]
    if drop_roles and isinstance(drop_roles, str):
        drop_roles = [drop_roles]
    if drop_roles:
        txts = []
        to_check = [f'[{role}]' for role in drop_roles]
        for uttn in text.split('\n'):
            if all(uttn.strip()[:len(i)] != i for i in to_check):
                txts.append(uttn)
            text = '\n'.join(txts)
    sents = split_sentence(text,
                           mark_noisy_sentence=True,
                           marker=_MEANINGLESS_TEXT,
                           simple=False)
    text = [sent for sent in sents if sent[:len(_MEANINGLESS_TEXT)] != _MEANINGLESS_TEXT]
    text = ' '.join(text)
    text = remove_roles(text)
    if use_soft_cutoff:
        text = soft_cutoff(text=text,
                           spoken_terms=spoken_terms,
                           wanted_like_sentences=wanted_like_sentences,
                           pos_thresh=pos_thresh,
                           unwanted_like_sentences=unwanted_like_sentences,
                           neg_thresh=neg_thresh,
                           model=model,
                           device=device)
    suffix = context["suffix"]
    if suffix is not None:
        text += suffix
    return text


CALLBACK_POOL = globals()
