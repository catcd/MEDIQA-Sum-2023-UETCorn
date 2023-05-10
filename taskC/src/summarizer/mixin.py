import inspect
import logging
from copy import copy
from typing import Union, Dict, Tuple, List, Any, Callable

import pandas as pd
import torch.cuda
import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from .callbacks import CALLBACK_POOL
from .util import *
from ..model_manager import get_bart_gen_model, get_question_answering_pipeline, auto_detect_device
from ..util import RetrievalResult

log = logging.getLogger(__file__)


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


def abs2ext_then_fill(retrieval_result: RetrievalResult,
                      question: str,
                      prefix: str,
                      model: str = "distilbert-base-cased-distilled-squad",
                      device: str = 'cpu',
                      return_score: bool = False,
                      normalize: bool = True,
                      **kwargs) -> Union[str, Tuple[str, float]]:
    pipeline = get_question_answering_pipeline(model)
    text = summarize(retrieval_result, normalize=normalize, device=device)
    answer = pipeline(question=question,
                      context=text, device=device, **kwargs)
    text_ans = prefix + answer['answer']
    if '{answer}' in prefix:
        text_ans = prefix.format(answer=answer['answer'])
    if return_score:
        return text_ans, answer['score']
    return text_ans


def summarize_each_group(retrieval_result: RetrievalResult,
                         model: Union[PreTrainedModel, str] = "philschmid/bart-large-cnn-samsum",
                         tokenizer: Union[PreTrainedTokenizer, str] = None,
                         prompt: str = None,
                         device: str = 'cpu',
                         use_auth_token=None,
                         normalize: bool = True,
                         **kwargs) -> str:
    results = []
    for text in retrieval_result.texts:
        each = summarize_text(text, model=model,
                              tokenizer=tokenizer,
                              prompt=prompt,
                              device=device,
                              use_auth_token=use_auth_token,
                              normalize=normalize,
                              **kwargs)
        results.append(each)
    return '\n\n'.join(results)


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


def naive_extract_after_terms_text(text: str,
                                   terms: Dict[str, list],
                                   prefix: str,
                                   normalize: bool = False,
                                   remove_role: bool = True,
                                   **kwargs):
    answers = {}
    for answer, t in terms.items():
        term_indexs = [text.find(sub_keyword) for sub_keyword in t]
        last_non_neg = [idx for idx in term_indexs if idx >= 0]
        if len(last_non_neg) > 0:
            last_non_neg = last_non_neg[-1]
            answers[answer] = text[last_non_neg:]
            if remove_role:
                answers[answer] = remove_roles(answers[answer])
            elif normalize:
                answers[answer] = normalize_to_usual_dialogue(answers[answer])
        else:
            answers[answer] = ''

    result = prefix.format(**answers)
    return result


def question_extract_then_fill_text(text: str,
                                    question: Union[List[str], str],
                                    prefix: Union[List[str], str] = None,
                                    model: str = "distilbert-base-cased-distilled-squad",
                                    device: str = 'cpu',
                                    return_score: bool = False,
                                    normalize: bool = True,
                                    replace_pronoun: bool = False,
                                    callback: Union[str, Callable] = None,
                                    callback_kwargs: dict = None,
                                    **kwargs) -> Union[Tuple[str, List[Any]], str]:
    pipeline = get_question_answering_pipeline(model)
    run_device = None
    if device == 'auto' and pipeline.device.type == 'cpu':
        run_device = auto_detect_device(pipeline)
    elif device != 'auto':
        run_device = device
    if run_device != None and pipeline.device != torch.device(run_device):
        pipeline.device = torch.device(run_device)
        pipeline.model.to(pipeline.device)
    else:
        run_device = pipeline.device

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
                                  context=text, device=run_device, **kwargs)
            except Exception as e:
                log.warning(f"Got exception {e}")
                log.warning(f"Retrying with CPU")
                answer = pipeline(question=question,
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


def naive_extract_after_terms(row: dict,
                              column: str,
                              suffix: str = None,
                              exceptions: List[str] = None,
                              callback: Union[str, Callable] = None,
                              callback_kwargs: dict = None,
                              **kwargs) -> str:
    text = row[column]
    substring2 = "-year-old"
    ext1, ext2 = "", ""
    r1, r2 = "", ""
    ""
    substring1 = [
        'assessment', 'my impression', ' plan'
    ]

    for s in substring1:
        if text.find(s) != -1:
            index = text.find(s)
            result = text[index:].strip()
            if exceptions:
                if any([result[:len(i)] == i for i in exceptions]):
                    continue
            result = "\n".join(result.split("\n<GROUP>\n"))
            r1 = result
            result = re.sub("(\\[doctor\\]|\\[patient\\])", "", result)
            ext1 = result

    new_text = text.split(".")
    for sent in new_text:
        if re.search(substring2, sent):
            result = sent
            result = "\n".join(result.split("\n<GROUP>\n"))
            r2 = result
            result = re.sub("(\\[doctor\\] |\\[patient\\] |\n<GROUP>\n)", "", result)
            ext2 = result

    ext = ext2 + "\n" + ext1
    if suffix is not None:
        ext += suffix
    if callback is not None:
        if isinstance(callback, str):
            callback = CALLBACK_POOL[callback]
        if callback_kwargs is None:
            callback_kwargs = {}
        context = {
            "suffix": suffix,
            "raw_extract": r2 + "\n" + r1,
            **kwargs
        }
        return callback(to_return=ext, context=context, **callback_kwargs)
    return ext


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
            log.warning(f"Got exception {e}")
            log.warning(f"Retrying with CPU")
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


def summarize(retrieval_result: RetrievalResult,
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
    text = '\n'.join(retrieval_result.texts)
    if not text.strip():
        return ''
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
        return do_format(prefix=prefix, ordered_answers=[non_prompt_summ], sep='')


def question_extract_then_fill(retrieval_result: RetrievalResult,
                               question: str,
                               prefix: str,
                               model: str = "distilbert-base-cased-distilled-squad",
                               device: str = 'cpu',
                               return_score: bool = False,
                               normalize: bool = True,
                               replace_pronoun: bool = False,
                               callback: Union[str, Callable] = None,
                               callback_kwargs: dict = None,
                               **kwargs) -> Union[Tuple[str, List[Any]], str]:
    text = '\n'.join(retrieval_result.texts)
    if not text.strip():
        return ''
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


def do_nothing(retrieval_result: RetrievalResult,
               prefix: str = None,
               callback: Union[str, Callable] = None,
               callback_kwargs: dict = None,
               **kwargs) -> str:
    result = '\n'.join(retrieval_result.texts)
    if not result.strip():
        return ''
    to_return = re.sub("(\\[doctor\\]|\\[patient\\])", "", result)
    if prefix != None:
        to_return = prefix + to_return
    if callback is not None:
        if isinstance(callback, str):
            callback = CALLBACK_POOL[callback]
        context = {
            'retrieval_result': retrieval_result,
            'prefix': prefix,
        }
        if callback_kwargs is None:
            callback_kwargs = {}
        return callback(to_return=to_return, context=context, **callback_kwargs)
    return to_return


def note_complete(df: pd.DataFrame,
                  dialogue_column: str,
                  index_column: str,
                  **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Make a complete note form retrieval results
    """
    import json

    assert dialogue_column in df.columns, f" Column '{dialogue_column}' not found in the given dataframe!"
    assert index_column in df.columns, f" Column '{index_column}' not found in the given dataframe!"
    df = df.rename(columns={index_column: 'encounter_id'})

    notes = []
    rows = df.to_dict(orient='records')
    for seeding_form, d_id, row in tqdm.tqdm(zip(df[dialogue_column].tolist(),
                                                 df['encounter_id'].tolist(),
                                                 rows),
                                             "Completing note..."):
        a_note = {'encounter_id': d_id}
        divisions = {}
        seeding_form = json.loads(seeding_form)
        for each_title_detail in seeding_form:
            assert 'division' in each_title_detail, \
                f"Not found 'division' variable in column {dialogue_column}. Current key {list(each_title_detail.keys())}"
            division = each_title_detail['division']
            assert division in ['subjective', 'objective_exam',
                                'objective_results', 'assessment_and_plan'], \
                f"Wrong division: '{division}'"
            if division not in divisions:
                divisions[division] = {}
            assert 'title' in each_title_detail, \
                f"Not found 'title' variable in column {dialogue_column}. Current key {list(each_title_detail.keys())}"
            title = each_title_detail['title']
            if title not in divisions[division]:
                divisions[division][title] = title.upper() + '\n'
            if 'summarizer' in each_title_detail:
                assert 'method' in each_title_detail['summarizer'], \
                    f"Not found 'method' variable in column 'summarizer'. Current key {list(each_title_detail['summarizer'].keys())}"
                method = globals()[each_title_detail['summarizer']['method']]

                if 'retrieval_result' in inspect.signature(method).parameters:
                    assert 'retrieval_result' in each_title_detail, \
                        f"Not found 'retrieval_result' variable in column {dialogue_column}"
                    retrieval_result = RetrievalResult(**each_title_detail['retrieval_result'])
                    other_kwargs = copy(each_title_detail['summarizer'])
                    del other_kwargs['method']
                    data = method(retrieval_result=retrieval_result, **other_kwargs)
                    if isinstance(data, str) and data.strip():
                        divisions[division][title] += '\n' + data
                elif 'row' in inspect.signature(method).parameters:
                    other_kwargs = copy(each_title_detail['summarizer'])
                    if 'column' not in other_kwargs:
                        assert 'fixed_role_dialogue' in row, \
                            "Expect 'fixed_role_dialogue' column is not in the input dataframe!"
                        log.warning('Not found config of summarizer:column in seeding_form '
                                    'while summarizer:method receives argument \'row\'. '
                                    'Using default \'fixed_role_dialogue\'')
                        other_kwargs['column'] = 'fixed_role_dialogue'
                    del other_kwargs['method']
                    data = method(row=row, **other_kwargs)
                    if isinstance(data, str) and data.strip():
                        divisions[division][title] += '\n' + data

        # for k, v in divisions.items():
        #     a_note[k] = "\n\n".join(v.values())

        total_note = []
        for division_name in ['subjective', 'objective_exam',
                              'objective_results', 'assessment_and_plan']:
            if division_name in divisions:
                div = "\n\n".join(divisions[division_name].values())
                total_note.append(div)
                a_note[division_name] = div
            else:
                log.warning(f"Division '{division_name}' not found in seeding_form!")
                a_note[division_name] = '---NONE---'

        a_note['note'] = "\n\n".join(total_note)
        notes.append(a_note)

    notes = pd.DataFrame(notes)
    return {'output': notes}
