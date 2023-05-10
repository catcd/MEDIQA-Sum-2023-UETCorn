import re
from typing import List

import numpy as np
import torch

from ..model_manager import get_sbert_model, auto_detect_device
from ..retrieval.text_extracting_v2 import cosine_matrix
from ..util import split_sentence

_MEANINGLESS_TEXT = '<--NOISY-->'


def build_markov(base_unit,
                 query,
                 model='paraphrase-MiniLM-L6-v2',
                 device='auto'):
    if isinstance(model, str):
        model = get_sbert_model(model)
    if device == 'auto' and model.device.type == 'cpu':
        device = auto_detect_device(model)
    elif device == 'auto':
        device = model.device
    else:
        device = torch.device(device)
    des_vec = model.encode(query, device=device,
                           show_progress_bar=False,
                           convert_to_numpy=True)
    if des_vec.ndim == 1:
        des_vec = des_vec.reshape(1, -1)
    base_embeds = model.encode(base_unit, device=device,
                               show_progress_bar=False,
                               convert_to_numpy=True)
    if base_embeds.ndim == 1:
        base_embeds = des_vec.reshape(1, -1)
    raw_markovw = cosine_matrix(des_vec, base_embeds)
    return raw_markovw


def soft_cutoff(text: str,
                spoken_terms: List[str] = None,
                wanted_like_sentences: List[str] = None,
                pos_thresh: float = None,
                unwanted_like_sentences: List[str] = None,
                neg_thresh: float = None,
                model: str = 'paraphrase-MiniLM-L6-v2',
                device: str = 'auto',
                **kwargs):
    sents = split_sentence(text, mark_noisy_sentence=True,
                           marker=_MEANINGLESS_TEXT,
                           simple=False)
    sents = [sent for sent in sents if sent[:len(_MEANINGLESS_TEXT)] != _MEANINGLESS_TEXT]
    if wanted_like_sentences and sents:
        a = build_markov(sents, wanted_like_sentences,
                         model=model, device=device)
        scores = a.max(axis=0)
        sents = [sents[i] for i in np.argwhere(scores >= pos_thresh).reshape(-1)]
    if unwanted_like_sentences and sents:
        a = build_markov(sents, unwanted_like_sentences,
                         model=model, device=device)
        scores = a.max(axis=0)
        sents = [sents[i] for i in np.argwhere(scores < neg_thresh).reshape(-1)]
    text = ' '.join(sents)
    if spoken_terms:
        pattern = f"\\b({'|'.join(spoken_terms)})" + r'[\s!"\#\$%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]'
        text = re.sub(pattern, "", text, flags=re.I)

    return text
