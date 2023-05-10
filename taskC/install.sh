#!/bin/sh

python -m venv --clear uetcorn_taskC_venv
# Python 3.8.5

# define color codes
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
RESET='\e[0m'
CURRENT_DIR=$(pwd)

#==========INSTALL GPU PACKAGES==============
. uetcorn_taskC_venv/bin/activate
pip install -U pip setuptools wheel
pip install nvidia-pyindex
pip install nvidia-tensorrt==7.2.3.4
python -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/uetcorn_taskC_venv/lib/python3.8/site-packages/tensorrt/:$(pwd)/uetcorn_taskC_venv/lib/python3.9/site-packages/tensorrt/:$(pwd)/uetcorn_taskC_venv/lib/python3.10/site-packages/tensorrt/" >> uetcorn_taskC_venv/bin/activate
deactivate

. uetcorn_taskC_venv/bin/activate
pip install -r requirements.txt

#==========VERIFY CUDA==============
# Run the PyTorch command and save the output to a file
num_cuda_devices=$(python -c "import torch; print(torch.cuda.device_count())")
# shellcheck disable=SC2039
echo -e "${GREEN}Number of CUDA devices detected by PyTorch: ${num_cuda_devices}${RESET}"

# Run the Tensorflow command and save the output to a file
tf_cuda_devices=$(python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))")
# shellcheck disable=SC2039
echo -e "${GREEN}List of CUDA devices detected by Tensorflow:${RESET}"
# shellcheck disable=SC2039
echo -e "${GREEN}${tf_cuda_devices}${RESET}"



#==========DOWNLOAD PRETRAINED HF MODEL==============

echo -e "${YELLOW}Downloading hf pretrained models${RESET}"
python -c "from sentence_transformers import SentenceTransformer; _ = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"
python -c "from transformers import pipeline; _ = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad', device='cpu'); _ = pipeline('ner','oliverguhr/fullstop-punctuation-multilang-large', grouped_entities=False, device='cpu')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; tokenizer = AutoTokenizer.from_pretrained('philschmid/bart-large-cnn-samsum'); model = AutoModelForSeq2SeqLM.from_pretrained('philschmid/bart-large-cnn-samsum')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; tokenizer = AutoTokenizer.from_pretrained('lidiya/bart-large-xsum-samsum'); model = AutoModelForSeq2SeqLM.from_pretrained('lidiya/bart-large-xsum-samsum')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; tokenizer = AutoTokenizer.from_pretrained('amagzari/bart-large-xsum-finetuned-samsum-v2'); model = AutoModelForSeq2SeqLM.from_pretrained('amagzari/bart-large-xsum-finetuned-samsum-v2')"
python -c "import nltk; nltk.download('punkt')"

deactivate
