# Build Image

## For devices with GPU

```bash
docker build -t deep-bioinfo-first-step-gpu . -f Dockerfile.gpu
```

## For devices without GPU

```bash
docker build -t deep-bioinfo-first-step-cpu . -f Dockerfile.cpu
```

pip install tiktoken blobfile protobuf

# Run Image

## For devices with GPU

```bash
docker run -it --rm -v $PWD:/workdir -w /workdir -p 8888:8888 --gpus all deep-bioinfo-first-step-gpu /bin/bash
```

## For devices without GPU

```bash
docker run -it --rm -v $PWD:/workdir -w /workdir -p 8888:8888 deep-bioinfo-first-step-cpu /bin/bash
```

# Prepare Sample Data

download sample data from [https://github.com/songlab-cal/tape](TAPE) and extract it to `data/fluorescence` directory.

```bash
sh download_data.sh
```
