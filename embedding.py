import transformers
from transformers import T5Model, T5Tokenizer, BertModel, BertTokenizer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import os


def get_embedding(model_name, sequences, batch_size=16):
    """
    Get embeddings of protein sequences using pretrained models.

    Args:
        model_name (str): Name of the pretrained model.
        sequences (list of str): List of protein sequences.
        batch_size (int): Batch size for inference.
    Returns:
        embeddings (np.ndarray): Array of embeddings with shape (n_sequences, n_model_dim).
    """
    if model_name in [
        "Rostlab/protbert",
        "Rostlab/prot_t5_xl_bfd",
        "Rostlab/prot_t5_xl_uniref50",
    ]:
        return get_embedding_ProtTrans(sequences, model_name, batch_size)
    elif model_name in [
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
    ]:
        return get_embedding_ESM(sequences, model_name, batch_size)


def get_embedding_ProtTrans(
    sequences, model_name="Rostlab/prot_t5_xl_bfd", batch_size=16
):
    """
    Get embeddings of protein sequences using ProtTrans models.

    Args:
        sequences (list of str): List of protein sequences.
        model_name (str): Name of the pretrained model.
        batch_size (int): Batch size for inference.
    Returns:
        embeddings (np.ndarray): Array of embeddings with shape (n_sequences, n_model_dim).
    """
    if model_name in ["Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50"]:
        model = T5Model.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif model_name in ["Rostlab/protbert"]:
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    max_len = max(len(seq) for seq in sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    if model_name in ["Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50"]:
        embeddings_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                batch_sequences = sequences[i : i + batch_size]
                inputs = tokenizer.batch_encode_plus(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs, decoder_input_ids=inputs["input_ids"])
                for j in range(len(batch_sequences)):
                    embeddings_list.append(
                        outputs.encoder_last_hidden_state[
                            j, inputs["attention_mask"][j]
                        ]
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                    )
            embeddings = np.stack(embeddings_list)
        return embeddings
    elif model_name in ["Rostlab/protbert"]:
        embeddings_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                batch_sequences = sequences[i : i + batch_size]
                inputs = tokenizer.batch_encode_plus(
                    batch_sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs)
                for j in range(len(batch_sequences)):
                    embeddings_list.append(
                        outputs.last_hidden_state[j, inputs["attention_mask"][j]]
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                    )
            embeddings = np.stack(embeddings_list)
        return embeddings


def get_embedding_ESM(sequences, model_name="esm2_t33_650M_UR50S", batch_size=16):
    """
    Get embeddings of protein sequences using ESM models.

    Args:
        sequences (list of str): List of protein sequences.
        model_name (str): Name of the pretrained model.
        batch_size (int): Batch size for inference.
    Returns:
        embeddings (np.ndarray): Array of embeddings with shape (n_sequences, n_model_dim).
    """
    if model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        embed_layer = 36
    elif model_name == "esm2_t48_15B_UR50D":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        embed_layer = 48
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        embed_layer = 33
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        embed_layer = 30
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        embed_layer = 12
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        embed_layer = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i : i + batch_size]

        ###
        # Codes below are from https://github.com/facebookresearch/esm and modified.
        seq_encoded_list = [alphabet.encode(seq_str) for seq_str in batch_sequences]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                len(batch_sequences),
                max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(alphabet.padding_idx)
        for i, seq_encoded in enumerate(seq_encoded_list):
            if alphabet.prepend_bos:
                tokens[i, 0] = alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(alphabet.prepend_bos) : len(seq_encoded)
                + int(alphabet.prepend_bos),
            ] = seq
            if alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(alphabet.prepend_bos)] = (
                    alphabet.eos_idx
                )
        batch_tokens = tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(
                batch_tokens, repr_layers=[embed_layer], return_contacts=False
            )
        token_embeddings = results["representations"][embed_layer].cpu().numpy()

        sequence_embeddings = []
        for i, seq_len in enumerate(batch_lens):
            sequence_embeddings.append(token_embeddings[i, 1 : seq_len - 1].mean(0))
        ###

        embeddings.append(np.stack(sequence_embeddings))
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def plot_embedding(
    embeddings, labels=None, title="", method="UMAP", show=False, save_path=None
):
    """
    Plot embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings with shape (n_sequences, n_model_dim).
        labels (np.ndarray): Array of labels with shape (n_sequences,).
        title (str): Title of the plot.
        method (str): Dimensionality reduction method. "PCA", "UMAP", or "tSNE".
        show (bool): Whether to show the plot.
        save_path (str): Path to save the plot.
    """
    if method == "PCA":
        pca = PCA(n_components=2)
        mapping = pca.fit_transform(embeddings)
    elif method == "UMAP":
        umap = UMAP(n_components=2)
        mapping = umap.fit_transform(embeddings)
    elif method == "tSNE":
        tsne = TSNE(n_components=2)
        mapping = tsne.fit_transform(embeddings)
    if labels is None:
        color = np.zeros(embeddings.shape[0])
        cmap = "tab10"
    else:
        color = labels
        cmap = "viridis"

    plt.figure(figsize=(10, 10))
    plt.scatter(mapping[:, 0], mapping[:, 1], c=color, cmap=cmap, alpha=0.5)
    if labels is not None:
        plt.colorbar()
    plt.title(title)
    if save_path is not None:
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_json("data/fluorescence/fluorescence_train.json", lines=False).rename(
        columns={"primary": "sequence"}
    )
    df["log_fluorescence"] = df["log_fluorescence"].map(lambda x: x[0])
    sequences = df["sequence"].tolist()
    model_name = "Rostlab/prot_t5_xl_bfd"
    embeddings = get_embedding(model_name, sequences)
    plot_embedding(
        embeddings,
        method="UMAP",
        save_path="UMAP.png",
        labels=df["log_fluorescence"].values,
    )
