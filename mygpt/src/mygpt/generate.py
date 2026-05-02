import torch
import torch.nn.functional as F


def generate(
    model: "GPT",
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively generate max_new_tokens after prompt_ids.

    Inputs:
        model:           a trained GPT (or compatible Module returning logits).
        prompt_ids:      long tensor of shape (B, T_prompt).
        max_new_tokens:  how many tokens to append.
        temperature:     softmax temperature; <1 sharpens, >1 flattens.
        top_k:           if given, restrict sampling to the top_k most-likely
                         tokens at each step.

    Output:
        long tensor of shape (B, T_prompt + max_new_tokens).
    """
    model.eval()
    ids = prompt_ids
    for _ in range(max_new_tokens):
        ids_cond = (
            ids[:, -model.max_seq_len :] if ids.shape[1] > model.max_seq_len else ids
        )
        with torch.no_grad():
            logits, _ = model(ids_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids
