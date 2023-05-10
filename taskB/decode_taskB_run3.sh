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
  prefix="v3-nonextra"
else
  # If it's set, use the value of the parameter
  prefix="$2"
fi


# Check if the file doesn't exist
if [ ! -f "cache/$prefix-cache.csv" ]; then
  # Execute the Python script

  # run preprocessing code
  python main.py -p "preprocessing" \
      -i $1 \
      -o "cache/$prefix-cache.csv" \
      -dc "dialogue" \
      -ic "ID" \
      -pc "config/config-v3-noextra.json"


else
  echo "Preprocessing file already exists."
fi



# run summarizing code
python main.py -p "summ-complete" \
      -i "cache/$prefix-cache.csv" \
      -o "run_results/$prefix-raw-results_run3.csv" \
      -ic "ID" \
      -hc "section_header" \
      -dc "dialogue" \
      -pc "config/config-v3-noextra.json"



# run postprocessing code
python main.py -i "run_results/$prefix-raw-results_run3.csv"   \
  -o "taskB_uetcorn_run3_mediqaSum.csv" \
  -dc "SystemOutput" \
  -ic "TestID" \
  -p "postprocessing" \
  -pc "config/config-v3-noextra.json"

#deactivate


