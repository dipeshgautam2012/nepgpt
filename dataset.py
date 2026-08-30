import re
import torch


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}
        self._build_vocab(text)

    def _build_vocab(self, text: str):
        ''' Builds vocabulary lookup tables from the input text corpus'''
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}

    def encode(self, text):
        """Converts a string of text into a list of integer indices based on the vocabulary."""
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        """Converts a list of integer indices back into a string of text based on the vocabulary."""
        return ''.join([self.itos[i] for i in indices])
    def num_tokens(self):
        """Returns the number of unique tokens in the vocabulary."""
        return self.vocab_size


class NepaliTokenizer:
    """Word and punctuation tokenizer for Unicode Devanagari text."""

    PUNCT_SET = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।॥")
    TOKEN_RE = re.compile(r"[^\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥]+|[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥]")

    def __init__(self, text: str):
        self.tokens = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}
        self._build_vocab(text)

    def tokenize(self, text: str):
        return self.TOKEN_RE.findall(text)

    def _build_vocab(self, text: str):
        self.tokens = sorted(set(self.tokenize(text)))
        self.vocab_size = len(self.tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos = {i: tok for i, tok in enumerate(self.tokens)}

    def encode(self, text):
        return [self.stoi[tok] for tok in self.tokenize(text)]

    def decode(self, indices):
        parts = []
        for i in indices:
            tok = self.itos[i]
            if parts and tok not in self.PUNCT_SET:
                parts.append(" ")
            parts.append(tok)
        return "".join(parts)

    def num_tokens(self):
        return self.vocab_size


class BatchSampler:
    """Samples sequence windows from a 1D token tensor.

    with_replacement=False: shuffled non-overlapping chunks, each used once per epoch.
    with_replacement=True: random start in 0 .. len(data)-block_size-1.
    """

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
        shuffle: bool = True, # shuffle is ignored if with_replacement is True
        with_replacement: bool = False,
    ):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.shuffle = shuffle
        self.with_replacement = with_replacement

    def __len__(self):
        # number of batches in the epoch (without replacement)
        n_chunks = len(self.data) // (self.block_size + 1)
        return n_chunks // self.batch_size  # e.g. 4 chunks, batch_size=2 → 2

    def __iter__(self):
        if not self.with_replacement:  # without replacement only
            self.cursor = 0
            chunk_len = self.block_size + 1
            n_chunks = len(self.data) // chunk_len
            # start indices of the chunks, used for without replacement only
            starts = torch.arange(n_chunks) * chunk_len  # n_chunks=4, block_size=5 → [0, 6, 12, 18]
            if self.shuffle:
                # shuffle the start indices
                starts = starts[torch.randperm(n_chunks)]  # e.g. [18, 0, 12, 6]
            
            self.starts = starts
        return self

    def __next__(self):
        if self.with_replacement:  # with replacement only
            # index of the start of the next batch
            # batch_size=4 → [3, 0, 8, 1]
            ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,)) 
            return self._windows(ix)

        # without replacement only
        end = self.cursor + self.batch_size
        if end > len(self.starts):
            raise StopIteration
        batch_starts = self.starts[self.cursor:end]  # batch_size=2 → first [0, 6], then [12, 18]
        self.cursor = end
        return self._windows(batch_starts)

    def _windows(self, starts):
        t = torch.arange(self.block_size)  # block_size=5 → [0, 1, 2, 3, 4]
        idx = starts.unsqueeze(1) + t  # batch_size=2, block_size=5: [0, 6] → [[0...4], [6...10]]
        x = self.data[idx]
        y = self.data[idx + 1]
        return x.to(self.device), y.to(self.device)


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).resolve().parent / "nepali_data" / "merged.txt"
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    sample = text[:200]
    ids = tokenizer.encode(sample)
    print(f"file: {path}")
    print(f"vocab size: {tokenizer.num_tokens()}")
    print(f"tokens in file: {len(tokenizer.encode(text))}")
    print(f"encoded: {ids[:20]}...")
    print(f"decoded: {tokenizer.decode(ids)}")

    # 1. Sample raw text
    sample_text = "First Citizen: Before we proceed any further, hear me speak. All: Speak, speak." * 100

    # 2. Tokenize text into a 1D long tensor
    chars = sorted(list(set(sample_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in sample_text], dtype=torch.long)

    # 3. Split 1D tensor upfront (90% train, 10% val)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # 4. Instantiate independent BatchSampler instances
    train_sampler = BatchSampler(train_data, batch_size=16, block_size=32, device="cpu", shuffle=True)
    val_sampler   = BatchSampler(val_data,   batch_size=16, block_size=32, device="cpu", shuffle=False)

    # 5. Get persistent training iterator
    train_iter = iter(train_sampler)

    # 6. Fetch a training batch
    try:
        x_train, y_train = next(train_iter)
    except StopIteration:
        train_iter = iter(train_sampler)
        x_train, y_train = next(train_iter)

    print(f"Train Inputs (X) shape: {x_train.shape} | Targets (Y) shape: {y_train.shape}")

    # 7. Fetch a validation batch
    x_val, y_val = next(iter(val_sampler))
    print(f"Val Inputs   (X) shape: {x_val.shape} | Targets (Y) shape: {y_val.shape}")