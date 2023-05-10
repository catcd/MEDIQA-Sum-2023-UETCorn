python main.py \
  -i $1 \
  --run "run2" \
  -o taskA_uetcorn_run2_mediqaSum.csv \
  -dc "dialogue" \
  -ic "ID" \
  --model_url "./pretrained_pipelines/binaryclass_model_v2.bin" \
  -pc "config.json"