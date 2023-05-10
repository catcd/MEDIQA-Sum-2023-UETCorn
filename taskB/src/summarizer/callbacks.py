from typing import Any, Union, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..model_manager import get_sbert_model, auto_detect_device
from ..util import split_sentence


def cosine_matrix(a: np.array, b: np.array) -> np.array:
    """ Calculate cosine matrix of two given matrix"""
    # normalize each vector in a and b
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = a / a_norm
    b_normalized = b / b_norm

    # calculate cosine similarity between each vector in a and b
    similarity = np.matmul(a_normalized, b_normalized.T)
    return similarity


def do_nothing(to_return: Any = None, context: dict = None, **callback_kwargs) -> str:
    return to_return


def drop_if_too_low(to_return: str,
                    evaluate_keywords: List[str] ,
                    thresh: float,
                    model: Union[str, SentenceTransformer] = 'paraphrase-MiniLM-L6-v2',
                    split_into_sentence: bool = True,
                    pooling_query_type: int = 'mean',
                    drop_whole: bool = False,
                    combine_type: int = 'mean',
                    device: str = 'auto',
                    **callback_kwargs) -> str:
    """
    thresh: cosine threshold to drop (SBERT)
    split_into_sentence: whether split whole answer to sentences or not
    pooling_query_type: query-pooling type
    drop_whole: whether drop whole example or keep important sentences
    combine_type: if drop_whole = True, how scores after query-pooling are combines
    """
    if isinstance(model, str):
        model = get_sbert_model(model)
    if device == 'auto' and not model.device.type != 'cuda':
        device = auto_detect_device(model)
    elif device == 'auto':
        device = model.device
    else:
        device = torch.device(device)

    if split_into_sentence:
        _sentence_texts = split_sentence(to_return, mark_noisy_sentence=False, simple=False)
    else:
        _sentence_texts = [to_return, ]
    des_vec = model.encode(evaluate_keywords,
                           device=device,
                           show_progress_bar=False,
                           convert_to_numpy=True)
    if des_vec.ndim == 1:
        des_vec = des_vec.reshape(1, -1)
    base_embeds = model.encode(_sentence_texts,
                               device=device,
                               show_progress_bar=False,
                               convert_to_numpy=True)
    if base_embeds.ndim == 1:
        base_embeds = base_embeds.reshape(1, -1)
    raw_markovw = cosine_matrix(des_vec, base_embeds)  # q x dim * dim x k = q x k
    assert pooling_query_type in ['sum', 'mean', 'max',
                                  'min'], f"pooling_query_type {pooling_query_type} not supported!"

    pool = dict(zip(['sum', 'mean', 'max', 'min'], [np.sum, np.mean, np.max, np.min]))
    query_pooled = pool[pooling_query_type](raw_markovw, axis=0)
    if drop_whole:
        assert combine_type in ['sum', 'mean', 'max',
                                'min'], f"combine_type {combine_type} not supported!"
        scalar_score = pool[combine_type](query_pooled, axis=0)
        if scalar_score >= thresh:
            return to_return
        return 'None'
    else:
        fine_sents_idxs = np.argwhere(query_pooled >= thresh).reshape(-1).tolist()
        fine_sents = [_sentence_texts[idx] for idx in fine_sents_idxs]
        if not fine_sents:
            return 'None'
        return ' '.join(fine_sents)


def drop_by_row_cate(to_return: str, row: dict, **callback_kwargs) -> str:
    """Demo"""
    if 'section_header' in row and row['section_header'] == 'CC':
        return 'None'
    return to_return


CALLBACK_POOL = globals()
