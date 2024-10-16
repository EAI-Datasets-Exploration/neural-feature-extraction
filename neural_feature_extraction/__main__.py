"""
Entry point into the neural feature extraction code which generates
embeddings for natural language commands.

run: 
$ python -m neural-feature-extraction
from neural-feature-extraction directory.
"""
import clip
import configparser
import json
import os
import pandas as pd
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import torch
import tensorflow as tf
from neural_feature_extraction.preprocessing import clean_and_standardize_text
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("../neural-feature-extraction/config.ini")

    model_name = config["experiment"]["model_name"]

    # Load Dataset
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    dataset_name = config["experiment"]["dataset_name"]

    with open(f"{dataset_dir_path}metadata.json", "r", encoding="utf-8") as json_file:
        # The metadata file referenced here is contained in the repo:
        # dataset-download-scripts package hosted in the larger GitHub group.
        metadata_dict = json.load(json_file)

    ds_path = metadata_dict[dataset_name]

    # Begin text feature processing
    df = pd.read_csv(dataset_dir_path + ds_path)
    df = clean_and_standardize_text(df, nl_column="nl_instructions")
    data = list(df["nl_instructions"].dropna().astype(str).unique())

    # Load other experiment parameters
    batch_size = config.getint("experiment", "batch_size")
    max_length = config.getint("experiment", "max_length")

    # Begin generating embeddings
    if model_name == "sbert":
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(data)
    elif model_name == "sonar":
        model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )
        embeddings = model.predict(data, source_lang="eng_Latn")
    elif model_name == "clip":
        filtered_sentences = [
            sentence for sentence in data if len(sentence.split()) <= max_length
        ]  # SCOUT has some really long utterances
        model, _ = clip.load("ViT-B/32")

        num_sentences = len(filtered_sentences)
        embedding_size = 512
        all_embeddings = torch.empty(
            (num_sentences, embedding_size), dtype=torch.float32, device="cuda"
        )

        for i in range(0, num_sentences, batch_size):
            batch = filtered_sentences[i : i + batch_size]
            text_tokens = clip.tokenize(batch).cuda()
            with torch.no_grad():
                batch_embeddings = model.encode_text(text_tokens).float()
            all_embeddings[i : i + batch_embeddings.size(0)] = batch_embeddings
        embeddings = all_embeddings.cpu()

    elif model_name == "use":
        num_sentences = len(data)
        model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

        all_embeddings = []
        for i in range(0, num_sentences, batch_size):
            subset = data[i : i + batch_size]
            embeddings = model(subset)
            embeddings_df = pd.DataFrame(embeddings.numpy())
            all_embeddings.append(embeddings_df)
        embeddings = pd.concat(all_embeddings, ignore_index=True)
    else:
        raise ValueError(f"Invalid method: {model_name}")

    if isinstance(embeddings, tf.Tensor):
        embeddings = embeddings.numpy()
    elif isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Save results
    results_dir = "results"
    results_fp = results_dir + "/" + model_name + "_test_" + dataset_name

    os.makedirs(results_dir, exist_ok=True)

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df["nl_command_exs"] = data
    embeddings_df.to_csv(f"{results_fp}.csv", index=False)
