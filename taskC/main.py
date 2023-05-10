import argparse
import inspect
import json
import logging
import os

import torch

from src import summarizer, preprocess, retrieval, postprocess
from src.util import read_dataset, save_taskC, save_df, read_dataframe

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))
log.debug("Starting ...")

parser = argparse.ArgumentParser()
current_dir = os.path.dirname(os.path.realpath(__file__))

pipelines = {
    "preprocessing": [read_dataset, preprocess.preprocessing, save_df],
    "create-reference": [read_dataset, save_taskC],
    "retrieval": [read_dataset, retrieval.form_retrieval, save_df],
    "retrieval-no-beam": [read_dataset, retrieval.no_BEAM_form_retrieval, save_df],
    "retrieval-beam-v2": [read_dataset, retrieval.BEAM_form_retrieval_v2, save_df],
    "summ:bart-no-prompt": [read_dataframe, summarizer.bart_summarizer, save_taskC],
    "summ:bart-prompt": [read_dataframe, summarizer.bart_summarizer_with_prompt, save_taskC],
    "note-complete": [read_dataframe, summarizer.note_complete, save_taskC],
    "postprocessing": [read_dataframe, postprocess.postprocessing, save_taskC],

}
current_support_pipelines = [f'{name} (task {task})' for task in pipelines for name in pipelines[task]]

parser.add_argument("--input_file_path", '-i', type=str, required=True,
                    help=f"Input file path (from {current_dir}). This must be a utf-8 csv file.")
parser.add_argument("--output_file_path", '-o', type=str, required=True,
                    help=f"Output file path (from {current_dir}).")
parser.add_argument("--dialogue_column", '-dc', type=str,
                    default="dialogue", help=f"The column of dialogue data in the given file.")
parser.add_argument("--index_column", '-ic', type=str,
                    default="encounter_id", help=f"The column of index in the given file.")
parser.add_argument("--pipeline", '-p',
                    required=True,
                    type=str, choices=[f'{name}' for name in pipelines],
                    help=f"The pipeline name. Current available is {current_support_pipelines}")
parser.add_argument("--config_file_path_for_pipeline", '-pc', type=str, required=False,
                    help="A json file that keeps the configuration of the current pipeline: "
                         "\n{[METHOD_NAME]:[KEY-VALUE-ARGUMENTS-FOR-THIS-METHOD]}")

args = parser.parse_args()

if __name__ == '__main__':
    log.debug("Loading config file ...")
    if args.config_file_path_for_pipeline is not None:
        with open(args.config_file_path_for_pipeline, 'r', encoding='utf-8') as f:
            pipe_config = json.load(f)
    else:
        pipe_config = {}

    pipeline = pipelines[args.pipeline]
    previous_output = {}
    sys_kwargs = vars(args)
    with torch.no_grad():
        for function in pipeline:
            log.debug(f"Executing function {function.__name__}....")
            kw = {}
            log.debug(inspect.signature(function).parameters)
            for param_name in inspect.signature(function).parameters:
                if param_name in sys_kwargs:
                    kw[param_name] = sys_kwargs[param_name]
            log.debug(kw)
            if function.__name__ in pipe_config:
                log.debug(f"Found arguments of {function.__name__} in the loaded configurations. "
                          f"Using these arguments")
                kw.update(pipe_config[f"{function.__name__}"])
            kw.update(previous_output)

            current_output = function(**kw)
            previous_output = current_output

