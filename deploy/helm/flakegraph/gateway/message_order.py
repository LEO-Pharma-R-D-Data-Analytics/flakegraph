"""Put every instruction message where the engine can read it.

The OpenAI chat contract lets a caller place ``system`` and ``developer``
messages anywhere in a conversation, and coding harnesses do exactly that when
they resume a session: the saved history carries their instruction updates
between turns. The engine's chat template merges instruction messages only
while they lead the conversation and refuses one that comes later, so a
resumed session fails with "System message must be at the beginning" while a
fresh one works.

The gateway is what advertises the OpenAI contract, so it is the gateway that
reconciles the two: instruction messages move to the front in the order they
were sent, and everything else keeps its order. Nothing is reworded, and a
conversation whose instructions already lead is passed through untouched.
"""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger

INSTRUCTION_ROLES = frozenset({"system", "developer"})
# The proxy names the route it is serving: chat completions carry the
# conversation in ``messages``, the Responses API carries it in ``input``
# (which may also be a bare string), and both admit instruction items anywhere.
CONVERSATION_FIELDS = {"completion": "messages", "acompletion": "messages", "aresponses": "input"}


def hoist_instructions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the messages with every instruction ahead of the conversation."""

    instructions = [m for m in messages if m.get("role") in INSTRUCTION_ROLES]
    if messages[: len(instructions)] == instructions:
        return messages
    return instructions + [m for m in messages if m.get("role") not in INSTRUCTION_ROLES]


class MessageOrder(CustomLogger):
    """Reorder a conversation before it reaches a model."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Hoist instruction items on the conversational routes; leave other calls alone."""

        field = CONVERSATION_FIELDS.get(call_type)
        if field is None:
            return data
        conversation = data.get(field)
        if isinstance(conversation, list) and all(isinstance(item, dict) for item in conversation):
            data[field] = hoist_instructions(conversation)
        return data


instance = MessageOrder()
