import src.preprocess as pp
from src.config import DEFAULT_MODEL_NAME
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
mda_dataset = pp.fetch_dataset('mda_dataset_imbalanced.joblib', 'mda_manual_v20251224_with_fold.csv')

token_len = [len(tokenizer.tokenize(text)) for text in tqdm(mda_dataset.data, desc="Calculating token lengths")]
token_len.remove(4)

max_len = max(token_len)
bins = [0, 512] + list(np.arange(1512, max_len + 1000, 1000))

plt.figure(figsize=(8, 5))
plt.hist(token_len, bins=bins, rwidth=0.95)

plt.axvline(
    x=512,
    color="red",
    linestyle="--",
    linewidth=1,
    label="Batas Maksimum BERT (512 token)"
)

xticks = [0] + list(np.arange(10000, max_len, 10000))
plt.xticks(sorted(set(xticks)))

plt.xlabel("Jumlah Token")
plt.ylabel("Jumlah Dokumen")
plt.title("Distribusi Panjang Dokumen")
plt.legend()
plt.savefig("histogram_token_length1.png", dpi=300, bbox_inches="tight")
plt.show()



