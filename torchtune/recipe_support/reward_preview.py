from typing import List, Sequence

import torch


def decode_responses(tokenizer, response_ids: torch.Tensor) -> List[str]:
    """
    Decode generated responses and normalize them so that thinking tags are always present.
    """
    decoded: List[str] = []
    response_ids = response_ids.detach().cpu()
    for row in response_ids:
        text = tokenizer.decode(row.tolist(), skip_special_tokens=False)
        stripped = text.lstrip()
        think_idx = stripped.find("<think>")
        if think_idx != -1:
            text = stripped[think_idx:]
        else:
            text = f"<think>{stripped}"
        decoded.append(text)
    return decoded


def decode_prompts(tokenizer, prompt_ids: torch.Tensor) -> List[str]:
    """
    Decode prompt tokens so we can optionally log them alongside previews.
    """
    prompt_ids = prompt_ids.detach().cpu()
    return [
        tokenizer.decode(row.tolist(), skip_special_tokens=False).strip()
        for row in prompt_ids
    ]


class RewardPreviewer:
    """
    Helper that prints raw GRPO completions at a configurable cadence.
    """

    def __init__(self, cfg, is_rank_zero: bool) -> None:
        cfg = cfg or {}
        count = int(cfg.get("count", 0) or 0)
        self._enabled = bool(count and is_rank_zero)
        self._max_examples = max(count, 0)
        self._char_limit = int(cfg.get("char_limit", 512))
        self._include_prompt = bool(cfg.get("include_prompt", False))
        self._include_target = bool(cfg.get("include_target", False))
        self._interval = int(cfg.get("weight_push_interval", 0) or 0)
        self._shown = 0
        self._tokenizer = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_tokenizer(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def maybe_reset(self, step: int) -> None:
        if not self._enabled:
            return
        if self._interval <= 0:
            # Only preview once unless explicitly reset.
            return
        if step % self._interval == 0:
            self._shown = 0

    def maybe_preview(
        self,
        prompt_ids: torch.Tensor,
        responses: List[str] | None,
        targets: Sequence[str],
        grpo_size: int,
    ) -> None:
        if not self._enabled or responses is None or not responses:
            return
        if self._tokenizer is None:
            return
        if self._shown >= self._max_examples:
            return

        prompts: List[str] | None = None
        if self._include_prompt:
            prompts = decode_prompts(self._tokenizer, prompt_ids)

        per_prompt = max(grpo_size, 1)
        for idx, response in enumerate(responses):
            if self._shown >= self._max_examples:
                break
            prompt_idx = min(idx // per_prompt, len(targets) - 1) if targets else 0
            prompt_text = prompts[prompt_idx] if prompts else None
            target_text = targets[prompt_idx] if targets else None

            self._shown += 1
            print(f"[Reward preview #{self._shown}]")
            if prompt_text is not None:
                print("Prompt:\n" + self._trim(prompt_text))
            print("Response before parsing:\n" + self._trim(response))
            if self._include_target and target_text is not None:
                print("Target:\n" + self._trim(target_text))
            print("-" * 40)

    def _trim(self, text: str) -> str:
        text = text.strip()
        if self._char_limit is None or self._char_limit < 0:
            return text
        limit = max(self._char_limit, 1)
        if len(text) <= limit:
            return text
        return text[:limit] + "... [truncated]"
