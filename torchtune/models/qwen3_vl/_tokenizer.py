# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional, Sequence

from torchtune.data import ChatMLTemplate, Message, truncate
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType
from torchtune.models.qwen3._tokenizer import Qwen3Tokenizer, QWEN3_SPECIAL_TOKENS
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json


class Qwen3VLTokenizer(Qwen3Tokenizer):  # noqa: N801
    """Tokenizer variant that inserts Qwen3 vision placeholders for image/video content."""

    def __init__(
        self,
        path: str,
        merges_file: str,
        special_tokens: dict[str, int],
        max_seq_len: Optional[int] = None,
        *,
        prompt_template=None,
        truncation_type: str = "right",
        spatial_merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            truncation_type=truncation_type,
            **kwargs,
        )
        self.vision_start_id = self.special_tokens.get("<|vision_start|>")
        self.vision_end_id = self.special_tokens.get("<|vision_end|>")
        self.image_token_id = self.special_tokens.get("<|image_pad|>")
        self.vision_pad_id = self.special_tokens.get("<|vision_pad|>")
        self.video_token_id = self.special_tokens.get("<|video_pad|>")
        self.spatial_merge_size = spatial_merge_size

    def _tokenize_header(self, messages, index: int) -> list[int]:
        tokens = []
        message = messages[index]
        if message.role == "ipython":
            if index == 0 or messages[index - 1].role != "ipython":
                self._add_message_start_tokens(tokens, "user")
                tokens.extend(self.encode("<tool_response>\n", add_bos=False, add_eos=False))
            else:
                tokens.extend(self.encode("\n<tool_response>\n", add_bos=False, add_eos=False))
        else:
            self._add_message_start_tokens(tokens, message.role)
            if message.role == "assistant" and message.ipython:
                tokens.append(self.tool_call_start_id)
                tokens.extend(self.encode("\n", add_bos=False, add_eos=False))
        return tokens

    def _tokenize_footer(self, messages, index: int) -> list[int]:
        tokens = []
        message = messages[index]
        if message.role == "ipython":
            if index == len(messages) - 1 or messages[index + 1].role != "ipython":
                tokens.extend(self.encode("\n</tool_response>", add_bos=False, add_eos=False))
                self._add_message_end_tokens(tokens)
            else:
                tokens.extend(self.encode("\n</tool_response>", add_bos=False, add_eos=False))
        else:
            if message.role == "assistant" and message.ipython:
                tokens.extend(self.encode("\n", add_bos=False, add_eos=False))
                tokens.append(self.tool_call_end_id)
            if message.role != "assistant" or index != len(messages) - 1:
                self._add_message_end_tokens(tokens)
        return tokens

    def _tokenize_message_content(self, message: Message) -> list[int]:
        tokens, _ = self._tokenize_message_content_with_grids(
            message=message,
            image_grid_iter=None,
            video_grid_iter=None,
        )
        return tokens

    def _tokenize_message_content_with_grids(
        self,
        *,
        message: Message,
        image_grid_iter: Optional[Iterator[Sequence[int]]],
        video_grid_iter: Optional[Iterator[Sequence[int]]],
    ) -> tuple[list[int], list[bool]]:
        tokens: list[int] = []
        visual_mask: list[bool] = []

        for item in message.content:
            if item["type"] == "text":
                text_tokens = self.encode(item["content"], add_bos=False, add_eos=False)
                tokens.extend(text_tokens)
                visual_mask.extend([False] * len(text_tokens))
            elif item["type"] == "image":
                if self.vision_start_id is not None:
                    tokens.append(self.vision_start_id)
                    visual_mask.append(False)
                if self.image_token_id is None:
                    raise RuntimeError("Image token id not found in tokenizer special tokens.")
                num_tokens = 1
                if image_grid_iter is not None:
                    try:
                        grid = next(image_grid_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Mismatch between provided image grids and message content."
                        ) from exc
                    t, h, w = (int(x) for x in grid)
                    num_tokens = (t * h * w) // (self.spatial_merge_size**2)
                tokens.extend([self.image_token_id] * num_tokens)
                visual_mask.extend([True] * num_tokens)
                if self.vision_end_id is not None:
                    tokens.append(self.vision_end_id)
                    visual_mask.append(False)
            elif item["type"] == "video":
                if self.vision_start_id is not None:
                    tokens.append(self.vision_start_id)
                    visual_mask.append(False)
                if self.video_token_id is None:
                    raise RuntimeError("Video token id not found in tokenizer special tokens.")
                num_tokens = 1
                if video_grid_iter is not None:
                    try:
                        grid = next(video_grid_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Mismatch between provided video grids and message content."
                        ) from exc
                    t, h, w = (int(x) for x in grid)
                    num_tokens = (t * h * w) // (self.spatial_merge_size**2)
                tokens.extend([self.video_token_id] * num_tokens)
                visual_mask.extend([True] * num_tokens)
                if self.vision_end_id is not None:
                    tokens.append(self.vision_end_id)
                    visual_mask.append(False)
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")
        return tokens, visual_mask

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_end_tokens: bool = True,
    ) -> tuple[list[int], list[bool]]:
        if isinstance(self.prompt_template, ChatMLTemplate):
            raise RuntimeError(
                "ChatMLTemplate is not supported for Qwen3VL tokenizer; use a different template or None."
            )

        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages: list[int] = []
        mask: list[bool] = []
        for idx, message in enumerate(templated_messages):
            tokens = self._tokenize_header(templated_messages, idx)
            content_tokens, content_visual_mask = self._tokenize_message_content_with_grids(
                message=message,
                image_grid_iter=None,
                video_grid_iter=None,
            )
            tokens.extend(content_tokens)
            tokens.extend(self._tokenize_footer(templated_messages, idx))

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokenized_messages.append(self.eos_id)
            mask.append(mask[-1] if mask else True)

        if self.max_seq_len:
            tokenized_messages = truncate(
                tokens=tokenized_messages,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_end_tokens else None,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_end_tokens else None,
                truncation_type=self.truncation_type,
            )

        return tokenized_messages, mask

    def tokenize_messages_with_grids(
        self,
        messages: list[Message],
        *,
        image_grid_thw: Optional[Sequence[Sequence[int]]] = None,
        video_grid_thw: Optional[Sequence[Sequence[int]]] = None,
        add_end_tokens: bool = True,
    ) -> tuple[list[int], list[bool], list[bool]]:
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages: list[int] = []
        mask: list[bool] = []
        visual_mask: list[bool] = []

        image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

        for idx, message in enumerate(templated_messages):
            header_tokens = self._tokenize_header(templated_messages, idx)
            tokenized_messages.extend(header_tokens)
            mask.extend([message.masked] * len(header_tokens))
            visual_mask.extend([False] * len(header_tokens))

            content_tokens, content_visual_mask = self._tokenize_message_content_with_grids(
                message=message,
                image_grid_iter=image_iter,
                video_grid_iter=video_iter,
            )

            tokenized_messages.extend(content_tokens)
            mask.extend([message.masked] * len(content_tokens))
            visual_mask.extend(content_visual_mask)

            footer_tokens = self._tokenize_footer(templated_messages, idx)
            tokenized_messages.extend(footer_tokens)
            mask.extend([message.masked] * len(footer_tokens))
            visual_mask.extend([False] * len(footer_tokens))

            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokenized_messages.append(self.eos_id)
            mask.append(mask[-1] if mask else True)
            visual_mask.append(False)

        if self.max_seq_len:
            tokenized_messages = truncate(
                tokens=tokenized_messages,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_end_tokens else None,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_end_tokens else None,
                truncation_type=self.truncation_type,
            )
            visual_mask = visual_mask[: len(tokenized_messages)]

        return tokenized_messages, mask, visual_mask


def qwen3_vl_tokenizer(
    path: str,
    merges_file: str,
    *,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
    spatial_merge_size: int = 2,
    **kwargs,
) -> Qwen3VLTokenizer:
    special_tokens = (
        QWEN3_SPECIAL_TOKENS
        if special_tokens_path is None
        else parse_hf_tokenizer_json(special_tokens_path)
    )

    template = (
        _get_prompt_template(prompt_template) if prompt_template is not None else None
    )

    return Qwen3VLTokenizer(
        path=path,
        merges_file=merges_file,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=template,
        truncation_type=truncation_type,
        spatial_merge_size=spatial_merge_size,
        **kwargs,
    )
