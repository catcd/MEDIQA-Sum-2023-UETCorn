import inspect
import logging
import re
from copy import deepcopy as copy
from typing import Union, Dict, Tuple, List, Any, Callable

import pandas as pd
import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .callbacks import CALLBACK_POOL
from ..model_manager import get_bart_gen_model, get_question_answering_pipeline, auto_detect_device

log = logging.getLogger(__file__)

_DEFAULT_COLUMN = 'clean_dialogue'


def normalize_to_usual_dialogue(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, r"\1:", text)
    return normalized_string


def remove_roles(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, '', text)
    return normalized_string


def do_format(prefix: str, ordered_answers: Union[List[str], str], sep: str = '\n') -> str:
    if prefix is None:
        if isinstance(ordered_answers, str):
            return ordered_answers
        return sep.join(ordered_answers)
    pattern = re.compile(r"\{(\w+)\}")
    matches = pattern.findall(prefix)
    matches.sort()
    if matches:
        if isinstance(ordered_answers, str):
            ordered_answers = [ordered_answers, ]
        ordered_answers += ['', ] * (len(matches) - len(ordered_answers))
        kv = dict(zip(matches, ordered_answers))
        return prefix.format(**kv)
    return prefix + sep.join(ordered_answers)


# Beta
def replace_pronouns(dialogue):
    new_dialogue = []
    pattern = r"\[(\w+)\]"
    roles = [re.sub(pattern, r"\1", m) for m in re.findall(pattern, dialogue)]
    for utt in dialogue.split('\n'):
        matches = re.findall(pattern, utt)

        if len(matches) > 0:
            normalized_role = re.sub(pattern, r"\1", matches[0])
            other_roles = [r for r in roles if r != normalized_role]
            has_other = len(other_roles) > 0
            i = normalized_role
            utt = re.sub(r"\bI\b", i, utt, flags=re.IGNORECASE)
            if has_other:
                # replace "you" to the first other role
                # warning: This may be in accurate
                utt = re.sub(r"\byou\b", other_roles[0], utt, flags=re.IGNORECASE)
        new_dialogue.append(utt)
    return '\n'.join(new_dialogue)


def summarize_text(text: str,
                   model: Union[PreTrainedModel, str] = "philschmid/bart-large-cnn-samsum",
                   tokenizer: Union[PreTrainedTokenizer, str] = None,
                   prompt: str = None,
                   device: str = 'auto',
                   use_auth_token=None,
                   normalize: bool = True,
                   return_best_only: bool = True,
                   num_beams: int = None,
                   callback: Union[str, Callable] = None,
                   callback_kwargs: dict = None,
                   **kwargs) -> Union[str, List[Tuple[str, float]]]:
    if isinstance(model, str):
        model, tokenizer = get_bart_gen_model(model_name=model, use_auth_token=use_auth_token)
    if device == 'auto' and not next(model.parameters()).is_cuda:
        device = auto_detect_device(model)
        model = model.to(device)
    elif device != 'auto':
        model = model.to(device)
    if normalize:
        text = normalize_to_usual_dialogue(text)
    encoder_inputs = tokenizer(text, truncation=True, return_tensors='pt')

    def _return(to_return, _callback, _callback_kwargs):
        if _callback is not None:
            if isinstance(_callback, str):
                _callback = CALLBACK_POOL[_callback]
            if _callback_kwargs is None:
                _callback_kwargs = {}
            context = {
                'text': text,
                'model': model,
                'tokenizer': tokenizer,
                'prompt': prompt,
                'device': device,
                'use_auth_token': use_auth_token,
                'normalize': normalize,
                'return_best_only': return_best_only,
                'num_beams': num_beams,
                **kwargs
            }
            return _callback(to_return=to_return, context=context, **_callback_kwargs)
        else:
            return to_return

    inputs = encoder_inputs
    if prompt:
        with tokenizer.as_target_tokenizer():
            decoder_inputs = tokenizer(prompt,
                                       truncation=True,
                                       return_tensors='pt',
                                       add_special_tokens=False)
            decoder_inputs = {f'decoder_{k}': v for k, v in decoder_inputs.items()}
        inputs.update(decoder_inputs)
    if num_beams is None:
        try:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen = model.generate(**inputs,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 **kwargs)
        except Exception as e:
            model = model.to('cpu')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen = model.generate(**inputs,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 **kwargs)

        gen_texts = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        if return_best_only:
            return _return(gen_texts[0], callback, callback_kwargs)
        else:
            text_x_score = [(txt, sc.cpu().numpy().tolist()) for txt, sc in zip(gen_texts, gen.sequences_scores)]
            return _return(text_x_score, callback, callback_kwargs)
    else:
        num_return_sequences = num_beams
        gen = model.generate(**inputs,
                             return_dict_in_generate=True,
                             num_return_sequences=num_return_sequences,
                             output_scores=True,
                             num_beams=num_beams,
                             **kwargs)
        gen_texts = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)

    if return_best_only:
        to_return = gen_texts[0]
    else:
        text_x_score = [(txt, sc.cpu().numpy().tolist()) for txt, sc in zip(gen_texts, gen.sequences_scores)]
        to_return = text_x_score

    return _return(to_return, callback, callback_kwargs)


def question_extract_then_fill_text(text: str,
                                    question: Union[List[str], str],
                                    prefix: Union[List[str], str],
                                    model: str = "distilbert-base-cased-distilled-squad",
                                    device: str = 'cpu',
                                    return_score: bool = False,
                                    normalize: bool = True,
                                    replace_pronoun: bool = False,
                                    callback: Union[str, Callable] = None,
                                    callback_kwargs: dict = None,
                                    **kwargs) -> Union[Tuple[str, List[Any]], str]:
    pipeline = get_question_answering_pipeline(model)
    run_device = pipeline.device
    if device == 'auto' and pipeline.device.type == 'cpu':
        run_device = auto_detect_device(pipeline)
    elif device != 'auto':
        run_device = device

    scores = None
    assert (normalize ^ replace_pronoun) or (not (
            normalize or replace_pronoun)), "Option invalid! " \
                                            "Only either 'replace_pronoun' or 'normalize', or none of them is turned on"
    if normalize:
        text = normalize_to_usual_dialogue(text)
    if replace_pronoun:
        text = replace_pronouns(text)
    if isinstance(question, str):  # single question
        try:
            answer = pipeline(question=question,
                              context=text, device=run_device, **kwargs)
        except Exception as e:
            log.warning(f"Got exception {e}")
            log.warning(f"Retrying with CPU")
            answer = pipeline(question=question,
                              context=text, device='cpu', **kwargs)
        text_ans = [answer['answer']]
        scores = [answer['score']]
    else:
        scores = []
        text_ans = []
        for each_question in question:
            try:
                answer = pipeline(question=each_question,
                                  context=text, device=device, **kwargs)
            except Exception as e:
                log.warning(f"Got exception {e}")
                log.warning(f"Retrying with CPU")
                answer = pipeline(question=each_question,
                                  context=text, device='cpu', **kwargs)
            scores.append(answer['score'])
            text_ans.append(answer['answer'])

    output = do_format(prefix=prefix, ordered_answers=text_ans, sep=' -- ')

    if return_score:
        to_return = (output, text_ans, scores)
    else:
        to_return = output
    if callback is not None:
        if isinstance(callback, str):
            callback = CALLBACK_POOL[callback]
        if callback_kwargs is None:
            callback_kwargs = {}
        context = {
            'text': text,
            'model': model,
            'device': device,
            'question': question,
            'prefix': prefix,
            'return_score': return_score,
            'normalize': normalize,
            'replace_pronoun': replace_pronoun,
            'text_ans': text_ans,
            'scores': scores,
            **kwargs
        }
        return callback(to_return=to_return, context=context, **callback_kwargs)

    return to_return


def question_extract_then_fill(row: dict,
                               column: str,
                               question: str,
                               prefix: str = None,
                               header_column: str = None,
                               section_header: str = None,
                               model: str = "distilbert-base-cased-distilled-squad",
                               device: str = 'cpu',
                               return_score: bool = False,
                               normalize: bool = True,
                               replace_pronoun: bool = False,
                               callback: Union[str, Callable] = None,
                               callback_kwargs: dict = None,
                               **kwargs) -> Union[Tuple[str, List[Any]], str]:
    if section_header is not None:
        if header_column is None or (header_column not in row) or row[header_column] != section_header:
            return ''
    text = row[column]
    if not text.strip():
        return ''
    if callback_kwargs is not None:
        callback_kwargs['row'] = row
    else:
        callback_kwargs = {'row': row}
    return question_extract_then_fill_text(text,
                                           question=question,
                                           prefix=prefix,
                                           model=model,
                                           device=device,
                                           return_score=return_score,
                                           normalize=normalize,
                                           replace_pronoun=replace_pronoun,
                                           callback=callback,
                                           callback_kwargs=callback_kwargs,
                                           **kwargs)


def summarize(row: dict,
              column: str,
              header_column: str = None,
              section_header: str = None,
              model: Union[PreTrainedModel, str] = "philschmid/bart-large-cnn-samsum",
              tokenizer: Union[PreTrainedTokenizer, str] = None,
              prompt: str = None,
              device: str = 'auto',
              use_auth_token=None,
              normalize: bool = True,
              num_beams: int = None,
              remove_prompt: bool = False,
              prefix: str = None,
              callback: Union[str, Callable] = None,
              callback_kwargs: dict = None,
              **kwargs) -> str:
    if section_header is not None:
        if header_column is None or (header_column not in row) or row[header_column] != section_header:
            return ''
    text = row[column]
    # print(text)
    if not text.strip():
        return ''
    if callback_kwargs is not None:
        callback_kwargs['row'] = row
    else:
        callback_kwargs = {'row': row}
    summ = summarize_text(text, model=model,
                          tokenizer=tokenizer,
                          prompt=prompt,
                          device=device,
                          use_auth_token=use_auth_token,
                          normalize=normalize,
                          return_best_only=True,
                          num_beams=num_beams,
                          callback=callback,
                          callback_kwargs=callback_kwargs,
                          **kwargs)
    if prompt is not None:
        non_prompt_summ = summ[len(prompt):]
    else:
        non_prompt_summ = summ
    if not remove_prompt:
        return summ
    else:
        return do_format(prefix=prefix, ordered_answers=[non_prompt_summ])


def summ_complete(df: pd.DataFrame,
                  index_column: str,
                  seeding_form: List[Dict],
                  header_column: str = None,
                  dialogue_column: str = None,
                  **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Make a complete note form retrieval results
    """

    # assert dialogue_column in df.columns, f" Column '{dialogue_column}' not found in the given dataframe!"
    assert index_column in df.columns, f" Column '{index_column}' not found in the given dataframe!"
    df = df.rename(columns={index_column: 'TestID'})
    notes = []
    rows = df.to_dict(orient='records')
    for d_id, row in tqdm.tqdm(zip(df['TestID'].tolist(), rows), "Completing summaries..."):
        a_note = {'TestID': d_id, 'SystemOutput': ''}
        for idx, each_title_detail in enumerate(seeding_form):
            assert 'summarizer' in each_title_detail, \
                f"Not found 'summarizer' variable in column seeding_from config number {idx}. Current key {list(each_title_detail.keys())}"
            assert 'method' in each_title_detail['summarizer'], \
                f"Not found 'method' variable in column 'summarizer'. Current key {list(each_title_detail['summarizer'].keys())}"
            try:
                method = globals()[each_title_detail['summarizer']['method']]
            except IndexError as e:
                log.warning(f"Ignore template because method {each_title_detail['summarizer']['method']} not found!")
                continue

            if 'row' in inspect.signature(method).parameters:
                other_kwargs = copy(each_title_detail['summarizer'])
                if dialogue_column is None:
                    if 'column' not in other_kwargs:
                        log.warning(f'Not found config of summarizer:column in seeding_form '
                                    'while summarizer:method receives argument \'row\'. '
                                    f'Using default \'{_DEFAULT_COLUMN}\'')
                        assert _DEFAULT_COLUMN in row, \
                            f"'{_DEFAULT_COLUMN}' not found in current dataframe!"
                        other_kwargs['column'] = _DEFAULT_COLUMN
                else:
                    other_kwargs['column'] = dialogue_column
                if header_column is not None and 'header_column' not in other_kwargs:
                    other_kwargs['header_column'] = header_column

                del other_kwargs['method']
                summ_output = method(row=row, **other_kwargs)
                a_note['SystemOutput'] += summ_output
            else:
                log.warning(f'Ignore template because "row" not in the signature of the method "{method.__name__}"')
        a_note['SystemOutput'] = a_note['SystemOutput'].strip()
        if not a_note['SystemOutput']:
            a_note['SystemOutput'] = 'None'
        notes.append(a_note)

    notes = pd.DataFrame(notes)
    return {'df': notes}
