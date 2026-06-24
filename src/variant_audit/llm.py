"""Provider-agnostic LLM access — THE key abstraction of the project.

Every LLM call in the codebase goes through `complete()`. It branches on
`settings.llm_provider` so you can develop locally on Ollama (free) and flip to
the Claude API for real eval runs by changing one env var. Keeping the swap
behind a single function is what makes that possible — don't call the SDKs
directly anywhere else.

TODO(day-1): implement the Ollama branch first (you only need this to start).
TODO(later): implement the Anthropic branch; wire token usage into telemetry.
"""
import logging

from .config import settings
from ollama import chat
from ollama import ChatResponse

logger = logging.getLogger(__name__)

def complete(prompt: str, system: str = "", *, purpose: str = "generate", max_tokens: int = 1024) -> str:
    """Return the model's text completion for a prompt.

    Args:
        prompt: the user message.
        system: optional system instruction.
        purpose: short tag for telemetry/cost attribution ("generate", "grade",
                 "groundedness", "eval_judge", ...). Also used to pick the judge model.
        max_tokens: output cap.

    Should branch on settings.llm_provider:
      - "ollama"    -> call the local Ollama server (see _complete_ollama)
      - "anthropic" -> call the Claude API (see _complete_anthropic)
    """ 
    raise NotImplementedError("TODO(day-1): route to the right provider and return text")


def _complete_ollama(prompt: str, system: str, model: str, max_tokens: int) -> str:
    """Call the local Ollama server. Hint: `import ollama; ollama.chat(...)`."""
    logger.info("ollama call model=%s max_tokens=%d", model, max_tokens)

    messages = []
    if system:
        messages.append({'role': "system", 'content': system})
    messages.append({'role': "user", 'content': prompt})

    try:
        response: ChatResponse = chat(
            model=model,
            messages=messages,
            options={'num_predict': max_tokens},
        )
    except Exception as e:
        logger.exception("ollama call failed model=%s", model)
        raise RuntimeError(f"Ollama call failed for model '{model}' — is `ollama serve` running?") from e

    return response['message']['content']


def _complete_anthropic(prompt: str, system: str, model: str, max_tokens: int) -> str:
    """Call the Claude API. Hint: `import anthropic; anthropic.Anthropic().messages.create(...)`."""
    raise NotImplementedError("TODO(later): call Claude, record token usage, return text")


def _model_for(purpose: str) -> str:
    """Pick the model: a cheaper 'judge' model for grading/judging, else the main model.

    This is where you save money on the API: grading/judging -> judge model (Haiku),
    classification reasoning -> main model (Sonnet).
    """
    raise NotImplementedError("TODO: return judge vs main model based on provider + purpose")
