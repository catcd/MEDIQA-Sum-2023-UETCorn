#!/bin/sh

#. ./uetcorn_taskC_venv/bin/activate

if [ ! -d "cache" ]; then
  mkdir "cache"
  echo "Directory created: cache"
else
  echo "Directory already exists: cache"
fi

if [ ! -d "run_results" ]; then
  mkdir "run_results"
  echo "Directory created: run_results"
else
  echo "Directory already exists: run_results"
fi

if [ -z "$2" ]; then
  # If it's not set, assign a default value
  prefix="config_v64"
else
  # If it's set, use the value of the parameter
  prefix="$2"
fi

# Check if the file doesn't exist
if [ ! -f "cache/$prefix-cache-v64.csv" ]; then
  # Execute the Python script

  # run preprocessing code
  python main.py -p "preprocessing" \
      -i $1 \
      -o "cache/$prefix-cache-v64.csv" \
      -dc "dialogue" \
      -ic "encounter_id" \
      -pc "config/bart-retrieval-v64.json"


else
  echo "Preprocessing file already exists."
fi



# run retrieval code
python main.py -p "retrieval-no-beam" \
      -i "cache/$prefix-cache-v64.csv" \
      -o "cache/$prefix-cache-v64.csv" \
      -dc "group_dialogue" \
      -ic "encounter_id" \
      -pc "config/bart-retrieval-v64.json"

# run summarizing code
python main.py -i "cache/$prefix-cache-v64.csv"  \
  -o "run_results/$prefix-taskC_uetcorn_run2_raw.csv" \
  -dc "retrieval_text" \
  -ic "encounter_id" \
  -p "note-complete" \
  -pc "config/bart-retrieval-v64.json"

# run postprocessing code
python main.py -i "run_results/$prefix-taskC_uetcorn_run2_raw.csv" \
  -o "taskC_uetcorn_run2_mediqaSum.csv" \
  -dc "note" \
  -ic "encounter_id" \
  -p "postprocessing" \
  -pc "config/bart-retrieval-v64.json"

#deactivate


