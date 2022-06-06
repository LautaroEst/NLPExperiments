from tokenizers.normalizers import Lowercase, Replace, Sequence as NormSeq
from tokenizers.pre_tokenizers import Whitespace, Sequence as PreTokSeq
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset



class HFCorpus(object):

    def __init__(self,batch_size,corpus_name,subdir,split,column):
        self._corpus = load_dataset(corpus_name,subdir,split=split)
        self.batch_size = batch_size
        self.column = column

    def __iter__(self):
        self.current = 0
        self.step = self.batch_size
        self.high = len(self._corpus)
        return self

    def __next__(self):
        current = self.current
        self.current += self.step
        if current < self.high:
            return self._corpus[current:current+self.step][self.column]
        else:
            raise StopIteration

    def __len__(self):
        return len(self._corpus)


_special_tokens = {
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
}

config = {

    # Is a pretrained tokenizer:
    "is_pretrained": False,

    # Tokenizer model:
    "model": WordLevel(
        vocab=None,
        unk_token=_special_tokens["unk_token"]
    ),

    # Normalizer:
    "normalizer": NormSeq([Lowercase(), Replace("10","diez")]),

    # Pre-tokenizer:
    "pre_tokenizer": Whitespace(),

    # Tokenizer training args:
    "trainer": WordLevelTrainer(
        vocab_size=30000,
        min_frequency=0,
        show_progress=True,
        special_tokens=list(_special_tokens.values())
    ),

    "corpus": HFCorpus(
        batch_size=32,
        corpus_name="amazon_reviews_multi",
        subdir="es",
        split="train",
        column="review_body"
    ),

    "encoding_args": {
        "model_max_length": 512,
        "padding_side": "right",
        "truncation_side": "right",
        "model_input_names": ["input_ids", "attention_mask"],
        "unk_token": _special_tokens["unk_token"],
        "bos_token": _special_tokens["bos_token"],
        "eos_token": _special_tokens["eos_token"],
        "sep_token": _special_tokens["sep_token"],
        "pad_token": _special_tokens["pad_token"],
        "cls_token": _special_tokens["cls_token"],
        "mask_token": _special_tokens["mask_token"],
        "additional_special_tokens": []
    }

}
