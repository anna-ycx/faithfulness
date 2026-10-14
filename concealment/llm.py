"""Send prompts through the repo's API helper and unpack the replies."""
import asyncio

import src.utils as utils
from src.templates.attrition import AttritionDataset
from src.utils import get_message, parse_response
from src.vllm_configs import VLLM_CONFIGS


async def _gather(prompts, model):
    # src.utils makes one semaphore at import; it binds to the first event loop that
    # uses it, so a second call_all in the same process would fail. Fresh one per batch.
    utils.semaphore = asyncio.Semaphore(utils.MAX_PARALLEL_REQUESTS)
    config = VLLM_CONFIGS[model]
    return await asyncio.gather(
        *[get_message(prompt=p, system_prompt="", config=config) for p in prompts],
        return_exceptions=True,
    )


def call_all(prompts, model):
    """One reply per prompt, in order; a failed call comes back as its exception."""
    return asyncio.run(_gather(prompts, model))


def unpack(response, valid_answers=AttritionDataset.VALID_ANSWERS) -> dict:
    """Pull the answer sections, hidden reasoning and call details out of one reply."""
    if isinstance(response, Exception) or "choices" not in response:
        return {"error": str(response)[:500]}
    choice, usage = response["choices"][0], response.get("usage", {})
    content = choice["message"].get("content") or ""
    parsed = parse_response(content, valid_answers) or {}
    return {
        "answer": parsed.get("answer"),
        "explanation": parsed.get("explanation"),
        "most_important_factors": parsed.get("most_important_factors"),
        "other_relevant_info": parsed.get("other_relevant_info"),
        "content": content,
        "reasoning": choice["message"].get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "provider": response.get("provider"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "error": None,
    }
