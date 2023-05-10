import argparse
import json
import logging
import os

import pandas as pd

from src.run1 import run1
from src.run2 import run2
from src.run3 import run3


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(os.path.basename(__file__))
log.debug("Starting ...")

parser = argparse.ArgumentParser()
current_dir = os.path.dirname(os.path.realpath(__file__))

runs = {
    'run1': run1,
    'run2': run2,
    'run3':run3
}

parser.add_argument("--input_file_path", '-i', type=str, required=True,
                    help=f"Input file path (from {current_dir}). This must be a utf-8 csv file.")
parser.add_argument("--output_file_path", '-o', type=str, required=True,
                    help=f"Output file path (from {current_dir}).")
parser.add_argument("--dialogue_column", '-dc', type=str,
                    default="dialogue", help=f"The column of dialogue data in the given file.")
parser.add_argument("--index_column", '-ic', type=str,
                    default="ID", help=f"The column of index in the given file.")
parser.add_argument("--config_file_path_for_pipeline", '-pc', type=str, required=False,
                    help="A json file that keeps the configuration of the current pipeline: "
                         "\n{[METHOD_NAME]:[KEY-VALUE-ARGUMENTS-FOR-THIS-METHOD]}")
parser.add_argument("--run", '-run', type=str, required=True,
                    choices=runs.keys(),
                    help=f"Available run: {list(runs.keys())}")
parser.add_argument("--model_url", '-murl', type=str, required=True,
                    help=f"Input pre-traind model path (from {current_dir}). This must be s pickle dumped file.")

args = parser.parse_args()

if __name__ == '__main__':
    method = runs[args.run]

    if args.config_file_path_for_pipeline is not None:
        log.debug("Loading config file ...")
        with open(args.config_file_path_for_pipeline, 'r', encoding='utf-8') as f:
            pipe_config = json.load(f)
    else:
        pipe_config = {}

    kwargs = {}
    if args.run in pipe_config:
        kwargs = pipe_config[args.run]

    output: pd.DataFrame = method(dataset_url=args.input_file_path,
                                  id_column=args.index_column,
                                  dialogue_column=args.dialogue_column,
                                  model_url=args.model_url,
                                  **kwargs
                                  )
    assert all([i in output.columns for i in ['TestID', 'SystemOutput']]), \
        f"Output does not contain either 'TestID' or 'SystemOutput' column! Found {output.columns}"
    log.info(f'Saving result to {args.output_file_path}')
    output.to_csv(args.output_file_path, columns=['TestID', 'SystemOutput'], index=False)
