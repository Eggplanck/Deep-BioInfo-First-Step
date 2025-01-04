# Deep Bioinfo First Step

This repository provides a first step example of bioinformatics using deep learning.

## Prerequisites

- Ubuntu (WSL2)
- Docker (In WSL2)
- NVIDIA Container Toolkit (For devices with GPU)

## Build Image

### For devices with GPU

```bash
docker build -t deep-bioinfo-first-step-gpu . -f Dockerfile.gpu
```

### For devices without GPU

```bash
docker build -t deep-bioinfo-first-step-cpu . -f Dockerfile.cpu
```

## Run Image

### For devices with GPU

```bash
docker run -it --rm -v $PWD:/workdir -w /workdir -p 8888:8888 --gpus all deep-bioinfo-first-step-gpu /bin/bash
```

### For devices without GPU

```bash
docker run -it --rm -v $PWD:/workdir -w /workdir -p 8888:8888 deep-bioinfo-first-step-cpu /bin/bash
```

## Prepare Sample Data

download sample data from [https://github.com/songlab-cal/tape](TAPE) and extract it to `data/fluorescence` directory.

```bash
sh download_data.sh
```

## Usage

### Embedding

You can get embedding of amino acid sequences using `embedding.get_embedding` function as follows.

```python
from embedding import get_embedding

sequence_list = [
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDERYK",
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "QNTPIGDGPVLLPDNHYLSAQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
    ]
model_name = "esm2_t6_8M_UR50D"

embedding = get_embedding(model_name, sequence_list) # tensor of shape (num_sequences, embedding_dim)
```

available model names are as follows.

- ESM2 (https://github.com/facebookresearch/esm)

  - "esm2_t36_3B_UR50D"
  - "esm2_t48_15B_UR50D"
  - "esm2_t33_650M_UR50D"
  - "esm2_t30_150M_UR50D"
  - "esm2_t12_35M_UR50D"
  - "esm2_t6_8M_UR50D"

- ProtTrans (published by Rostlab in Hugging Face https://huggingface.co/Rostlab)
  - "Rostlab/protbert"
  - "Rostlab/prot_t5_xl_bfd"
  - "Rostlab/prot_t5_xl_uniref50"

### Function Prediction

You can predict functions of amino acid sequences using `function_prediction.predict` function from embedding as follows.

```python
from function_prediction import get_predictor

training_sequence_list = [
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDERYK",
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "QNTPIGDGPVLLPDNHYLSAQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
]
training_function_score_list = [3.8237, 3.7520, 3.5401]

validation_sequence_list = [
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "QNTPIGDGPVLLPDNHILSTQSALSKDPNEKRDHLVLLTFVTAAGITHGMDELYK",
    "QNTPIGDGPVALPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
]
validation_function_score_list = [3.6604, 3.1145, 3.8502]

test_sequence_list = [
    "QNTPIGDGYVLLPDNHYLSTQSALSKDPNEKRDHMVLREFVTAAGITHGMDELYK",
    "QNAPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVAAAGITLGMDELYK",
]

model_name = "esm2_t6_8M_UR50D"
embedding = get_embedding(model_name, training_sequence_list + validation_sequence_list + test_sequence_list) # Embedding of all sequences

training_embedding = embedding[:len(training_sequence_list)]
validation_embedding = embedding[len(training_sequence_list):len(training_sequence_list)+len(validation_sequence_list)]
test_embedding = embedding[len(training_sequence_list)+len(validation_sequence_list):]


predictor, validation_loss = get_predictor(training_embedding, training_function_score_list, validation_embedding, validation_function_score_list) # Get predictor and validation loss
test_prediction = predictor(test_embedding) # Predict function scores of test sequences
```
