import pandas as pd

def extract_embeddings(fp: str, col_name="embedding"):
    df = pd.read_csv(fp)
    embeddings = df[col_name].to_list()
    return [[float(x.strip(" []")) for x in s.split(",")] for s in embeddings]




