import itertools

from echochain.client import chat
from echochain.config import MAX_CYCLES, ROTATING_MODELS, STOP_SCORE
from echochain.judge import score
from echochain.prompts import A_TO_Q, Q_TO_A


def cycle(seed: str, recorder) -> None:
    current = seed
    original = seed
    models = itertools.cycle(ROTATING_MODELS)
    avg = 10.0
    for idx in range(1, MAX_CYCLES + 1):
        phase = "A→Q" if idx % 2 else "Q→A"
        model = next(models)
        prompt = (
            A_TO_Q.render(answer=current)
            if phase == "A→Q"
            else Q_TO_A.render(question=current)
        )
        messages = [{"role": "user", "content": prompt}]
        current = chat(model, messages, temperature=0.3)
        sc = score(original, current, n=1)

        recorder.log(cycle=idx, phase=phase, model=model, text=current, score=sc)
        avg = ((avg * (idx - 1)) + sc) / idx
        if avg <= STOP_SCORE:
            break
