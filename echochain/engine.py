from __future__ import annotations

import itertools

from echochain.client import ChatClient, ChatMessage
from echochain.config import MAX_CYCLES, ROTATING_MODELS, STOP_SCORE
from echochain.judge import score
from echochain.persistence import CycleRecord, Recorder
from echochain.prompts import A_TO_Q, Q_TO_A

_INITIAL_AVERAGE = 10.0
_TRANSFORM_TEMPERATURE = 0.3


def cycle(seed: str, recorder: Recorder, client: ChatClient) -> None:
    """Run telephone cycles until the stop score or the cycle limit.

    Args:
        seed: Original answer.
        recorder: Sink for each cycle.
        client: Chat client used for transformations and judging.
    """
    current = seed
    original = seed
    models = itertools.cycle(ROTATING_MODELS)
    average = _INITIAL_AVERAGE
    for index in range(1, MAX_CYCLES + 1):
        phase = "A→Q" if index % 2 else "Q→A"
        model = next(models)
        prompt = (
            A_TO_Q.render(answer=current)
            if phase == "A→Q"
            else Q_TO_A.render(question=current)
        )
        messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
        reply = client.chat(model, messages, temperature=_TRANSFORM_TEMPERATURE)
        current = reply.text
        scored = score(original, current, client, n=1)
        row: CycleRecord = {
            "cycle": index,
            "phase": phase,
            "model": reply.requested_model,
            "served_model": reply.served_model,
            "text": current,
            "score": scored,
        }
        recorder.log(row)
        average = ((average * (index - 1)) + scored) / index
        if average <= STOP_SCORE:
            break
