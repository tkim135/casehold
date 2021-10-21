from collections import defaultdict
from tokenizers import Tokenizer, Encoding
import pathlib
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple
)

class BaseTokenizer(object):
    """Inherit from this ABC to build a CT2 tokenizer.
    """

    def __call__(self, text: Union[str, List[str]], *args, **kwargs) -> Dict[str, Any]:
        pass

    def __len__(self) -> int:
        """Vocab size.
        """
        pass

    def md5(self) -> str:
        """Hash summarising configuration of an instance.
        """

class SN20211005Tokenizer(BaseTokenizer):

    def __init__(self,
                 vocab_file : str,
                 name: Union[str, pathlib.Path] = 'gpt',
                 version: Optional[str] = '2.0.0',
                 unk_token: str = '<|unktoken|>',
                 eos_token: str = '\n\n',
                 pad_token: Optional[str] = '<|padtoken|>',  # If None do not pad.
                 pad_direction: str = 'right',
                 pad_to_length: int = 128,
                 ignore_cached: bool = False):

        pad_to_length = 128 # TODO: clean this, only doing because it's None otherwise
        path_to_tokenizer = pathlib.Path(vocab_file) # / f"{name}.{version}.json"
        self._tokenizer = Tokenizer.from_file(path_to_tokenizer.as_posix())

        self._name = name
        self._version = version
        self._unk_token = unk_token
        self._eos_token = eos_token
        self._pad_token = pad_token

        self._pad_direction = pad_direction
        self._pad_to_length = pad_to_length

        special_tokens = [unk_token, eos_token] 
        if pad_token is not None:
            special_tokens.append(pad_token)
        self.add_tokens(special_tokens, special_tokens=True)

        # configure truncation and padding
        target = {"max_length": pad_to_length, "stride": 0, "strategy": "longest_first"}
        if self._tokenizer.truncation != target:
            self._tokenizer.enable_truncation(**target)
        self._configure_padding()

    def __call__(self, text: Union[str, List[str]], pair: Optional[str] = None) -> List[Encoding]:
        if pair:
            if isinstance(text, str):
                text = [(text, pair)]
        else:
            if isinstance(text, str):
                text = [text]

        encodings = self._tokenizer.encode_batch(text, add_special_tokens=True, is_pretokenized=False)
        if not pair:
            return encodings

        # borrowed from PreTrainedTokenizerFast _batch_encode_plus
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_overflowing_tokens=True,
            )
            for encoding in encodings
        ]

        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        return sanitized_tokens

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        str_repr = f"""
            {self.__class__.__name__}
             name:{self._name}
             version:{self._version}
             unk_token:{self._unk_token}
             eos_token:{self._eos_token}
             pad_token:{self._pad_token}
             pad_direction:{self._pad_direction}
             pad_to_length:{self._pad_to_length}
        """
        return str_repr

    @property
    def md5(self) -> str:
        encoder = json.dumps(self.vocab, sort_keys=True)
        config = f"{str(self)}:{encoder}"
        return hashlib.md5(config.encode('utf-8')).hexdigest()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def pad_token_id(self) -> Optional[int]:
        if self._pad_token is None:
            return None
        return self.token_to_id(self._pad_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self._eos_token)

    def add_tokens(self, tokens: List[str], special_tokens: bool = False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(tokens)
        return self._tokenizer.add_tokens(tokens)

    def token_to_id(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            index = self._tokenizer.token_to_id(self._unk_token)
        return index

    def id_to_token(self, idx: int) -> str:
        return self._tokenizer.id_to_token(idx)
   
    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens
    def convert_tokens_to_ids(self, input_tokens):
        ids = []
        for tokens in input_tokens:
            ids.append(self.token_to_id(tokens))
        return ids
    def _configure_padding(self):
        if not self._pad_token:
            self._tokenizer.no_padding()
            return

        self._tokenizer.enable_padding(
            length=self._pad_to_length,
            direction=self._pad_direction,
            pad_id=self.pad_token_id,
            pad_type_id=0,
            pad_token=self._pad_token,
            pad_to_multiple_of=self._pad_to_length
        )

    # borrowed from transformers.tokenization_utils_fast.PreTrainedTokenizerFast
    def _convert_encoding(
        self,
        encoding: Encoding,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[Encoding]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = False
        if return_attention_mask is None:
            return_attention_mask = True

        encodings = [encoding]

        encoding_dict = defaultdict(list)
        if len(encodings) == 1:
            encoding_dict["input_ids"] = encodings[0].ids
            if return_token_type_ids:
                encoding_dict["token_type_ids"] = encodings[0].type_ids
            if return_attention_mask:
                encoding_dict["attention_mask"] = encodings[0].attention_mask
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"] = encodings[0].special_tokens_mask
            if return_offsets_mapping:
                encoding_dict["offset_mapping"] = encodings[0].offsets
            if return_length:
                encoding_dict["length"] = len(encodings[0].ids)
        else:
            for e in encodings:
                encoding_dict["input_ids"].append(e.ids)

                if return_token_type_ids:
                    encoding_dict["token_type_ids"].append(e.type_ids)
                if return_attention_mask:
                    encoding_dict["attention_mask"].append(e.attention_mask)
                if return_special_tokens_mask:
                    encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
                if return_offsets_mapping:
                    encoding_dict["offset_mapping"].append(e.offsets)
                if return_length:
                    encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings
