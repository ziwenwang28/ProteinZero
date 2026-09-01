from typing import List, Optional, Union
from transformers import PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy, TensorType
import torch

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

class iPLMTokenizer(PreTrainedTokenizerFast):
    def __init__(self, n_queries, use_structure=True, parallel=False, **kwargs):
        # Resolve the tokenizer specification when tokenizer_file is not provided explicitly.
        tok_file = kwargs.get('tokenizer_file')
        if tok_file is None:
            import os as _os
            name_or_path = kwargs.get('name_or_path', '')
            if name_or_path and _os.path.isdir(name_or_path):
                candidate = _os.path.join(name_or_path, 'tokenizer.json')
                if _os.path.isfile(candidate):
                    tok_file = candidate
            if tok_file is None:
                raise FileNotFoundError(
                    'tokenizer_file not found. Pass tokenizer_file= explicitly or ensure '
                    'tokenizer.json exists in the model directory.'
                )
        super().__init__(tokenizer_object=create_tokenizer_custom(tok_file), **kwargs)
        self.add_special_tokens({'pad_token': '<|pad|>'})
        self.use_structure = use_structure
        self.n_queries = n_queries if use_structure else 0
        self.parallel = parallel

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:

        if not isinstance(text, list):
            text = [text]

        # Allocate prefix tensors when structure conditioning is enabled.
        if self.use_structure:
            attn_mask_prefix = torch.zeros((len(text), self.n_queries), dtype=torch.bool)
            input_ids_prefix = torch.zeros((len(text), self.n_queries), dtype=torch.int)

        raw_text = []
        for i in range(len(text)):
            if '|' in text[i]:
                # Expected input format: "structure_identifier|protein_sequence".
                parts = text[i].split('|')
                raw_text.append(parts[1])  # Retain the protein sequence.

                if self.use_structure:
                    # Encode the structure identifier as character code points.
                    struct_id = torch.tensor([ord(c) for c in parts[0]])
                    length_id = len(struct_id)
                    input_ids_prefix[i, :length_id] = struct_id
                    attn_mask_prefix[i] = True
            else:
                raw_text.append(text[i])

        # Pass tokenizer arguments by keyword to preserve the tokenizer call signature.
        batch = super().__call__(
            text=raw_text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )

        # Prepend structure-identifier tokens and the corresponding attention mask.
        if self.use_structure:
            batch["attention_mask"] = torch.cat([attn_mask_prefix, batch["attention_mask"]], dim=1)
            batch["input_ids"] = torch.cat([input_ids_prefix, batch["input_ids"]], dim=1)

        # Remove token-type IDs returned by the underlying tokenizer.
        if "token_type_ids" in batch:
            del batch["token_type_ids"]

        return batch
