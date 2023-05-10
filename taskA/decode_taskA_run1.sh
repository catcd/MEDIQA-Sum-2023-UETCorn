python main.py \
  -i $1 \
  -o taskA_uetcorn_run1_mediqaSum.csv \
  -dc "dialogue" \
  -ic "ID" \
  --run "run1" \
  --model_url "./pretrained_pipelines/multiclass_model_v2.bin" \
  -pc "config.json"