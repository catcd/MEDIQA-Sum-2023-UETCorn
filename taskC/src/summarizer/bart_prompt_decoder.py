import logging
from typing import List, Dict

import pandas as pd
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

log = logging.getLogger(__file__)

from ..model_manager import get_bart_gen_model


def summarize_text_with_prompt(texts: List[str],
                               prompts: List[str],
                               model: PreTrainedModel,
                               tokenizer: PreTrainedTokenizer,
                               device: str,
                               **kwargs) -> List[str]:
    """
    texts: input text for summarization without prompting
    model: generative pretrained model (on HuggingFace)
    tokenizer: pretrained tokenizer
    **kwargs: other key-value arguments for generating process (see argument of PreTrainedModel.generate)
    this function return a Dataframe of ['encounter_id', 'note']
    """

    assert len(texts) > 0, "Empty input!"

    # ========== Phrase key-value argument =============
    model = model.to(device)

    # ========== Generate ===============================
    generated_texts = []
    from tqdm import tqdm
    for text, prompt in tqdm(zip(texts, prompts), desc="Summarizing with prompt decoder..."):
        encoder_inputs = tokenizer(text, truncation=True, return_tensors='pt')
        encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}
        inputs = encoder_inputs
        if prompt:
            with tokenizer.as_target_tokenizer():
                decoder_inputs = tokenizer(prompt,
                                           truncation=True,
                                           return_tensors='pt',
                                           add_special_tokens=False)
            decoder_inputs = {f'decoder_{k}': v.to(device) for k, v in decoder_inputs.items()}
            inputs.update(decoder_inputs)
        gen = model.generate(**inputs, **kwargs)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        generated_texts.append(gen)

    return generated_texts


def bart_summarizer_with_prompt(df: pd.DataFrame,
                                model_name: str,
                                dialogue_column: str,
                                index_column: str,
                                use_auth_token: str = None,
                                **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Summarize from snippets. Order of retrieval texts is IMPORTANT to reconstruct the note
    """
    import json

    assert dialogue_column in df.columns, f" Column '{dialogue_column}' not found in the given dataframe!"
    assert index_column in df.columns, f" Column '{index_column}' not found in the given dataframe!"
    df = df.rename(columns={index_column: 'encounter_id'})
    key_snip = []
    text_2_summ = []
    prompts = []
    for dialogue_retrieval_results, d_id in zip(df[dialogue_column].tolist(),
                                                df['encounter_id'].tolist()):
        for section, query, prompt_decoder, support_texts, scores in json.loads(dialogue_retrieval_results):
            context = support_texts[0]
            context = context.replace('[', '').replace(']', ':')
            key_snip.append((d_id, section))  # order to reconstruct the note
            text_2_summ.append(context)
            prompts.append(prompt_decoder)

    model, tokenizer = get_bart_gen_model(model_name=model_name, use_auth_token=use_auth_token)
    summaries = summarize_text_with_prompt(text_2_summ,
                                           prompts=prompts,
                                           model=model,
                                           tokenizer=tokenizer,
                                           **kwargs)
    notes = []
    _sec = None
    for (d_id, section), summary in zip(key_snip, summaries):
        if len(notes) > 0 and notes[-1]['encounter_id'] == d_id:
            # append to previous note
            if _sec == section:
                # merge to previous section
                notes[-1]['note'] += '\n' + summary
            else:
                notes[-1]['note'] += '\n' + section.upper() + '\n' + summary
        else:
            # new id
            notes.append({
                'encounter_id': d_id,
                'note': section.upper() + '\n' + summary
            })
        _sec = section
    notes = pd.DataFrame(notes)
    return {'output': notes}
