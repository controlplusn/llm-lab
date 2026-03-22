import os
import json
import tempfile
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


class BPEtokenizer:
    
    def __init__(self):
        self.tokenizer = None


    def train(self, corpus_files: list, vocab_size: int, allowed_special={"<|endoftext|>"}):
        self.tokenizer = ByteLevelBPETokenizer()

        self.tokenizer.train(
            files=corpus_files,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(allowed_special)
        )

        os.makedirs("tokenizer/", exist_ok=True)
        self.tokenizer.save_model("tokenizer/")
        print(f"Tokenizer trained. Vocab size: {vocab_size}")


    def load(self):
        self.tokenizer = ByteLevelBPETokenizer(
            vocab="tokenizer/vocab.json",
            merges="tokenizer/merges.txt"
        )


    def tokenize(self, text: str) -> list:
        return self.tokenizer.encode(text).tokens

    
    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids
    

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)


def stream_dataset_to_file(hf_dataset: str, sample_size: int = 200_000) -> str:
    print(f"Streaming {sample_size} rows from {hf_dataset}...")

    dataset = load_dataset(hf_dataset, split="train", streaming=True)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    )

    count = 0
    for row in dataset:
        line = row.get("text", "")
        if line.strip():
            tmp.write(line.strip() + "\n")
            count += 1
        if count >= sample_size:
            break

    tmp.close()
    print(f"Written {count} rows to temp file: {tmp.name}")
    return tmp.name


def main():
    tokenizer = BPEtokenizer()
    HF_DATASET = "anaoly/fineweb-cleaned-20B"

    if not os.path.exists("tokenizer/vocab.json"):
        print("Training tokenizer...")

        tmp_path = stream_dataset_to_file(HF_DATASET, sample_size=200_000)

        try:
            tokenizer.train(
                corpus_files=[tmp_path],
                vocab_size=32_000
            )
            print("Tokenizer training complete.")
        finally:
            os.remove(tmp_path)
            print("Temp file deleted.")
    else:
        print("Tokenizer already trained. Skipping...")


    tokenizer.load()
    print("Tokenizer loaded.")

    sample = "the cat sat on the mat."
    token = tokenizer.tokenize(sample)
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)


    print(f"\nSample: {sample}")
    print(f"Tokenize: {token}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")

    assert decoded == sample, "Encode/decode mismatch"



if __name__ == "__main__":
    main()