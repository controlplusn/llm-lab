from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

VOCAB_PATH  = "vocab.json"
MERGES_PATH = "merges.txt"
OUTPUT_PATH = "tokenizer.json"

def build():
    print("Building tokenizer...")
 
    tokenizer = Tokenizer(BPE.from_file(VOCAB_PATH, MERGES_PATH))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.save(OUTPUT_PATH)
 
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
 
    # Quick sanity check
    sample = "The model learns to predict the next token."
    encoded = tokenizer.encode(sample)
    print("\nSanity check:")
    print(f"Input: '{sample}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
 
 
if __name__ == "__main__":
    build()