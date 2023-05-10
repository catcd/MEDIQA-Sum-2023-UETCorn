python main.py \
  -i $1 \
  -o taskA_uetcorn_run3_mediqaSum.csv \
  -dc "dialogue" \
  -ic "ID" \
  --run "run3" \
  --model_url "./pretrained_pipelines/multiclass_model_f4.bin" \
  -pc "config.json"