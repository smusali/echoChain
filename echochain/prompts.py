from jinja2 import Template


A_TO_Q = Template("""
Given the following statement, formulate a concise technical question for which this
statement is the direct and most appropriate answer.
Do not be conversational. Output only the question.

Statement: "{{ answer }}"
Question:
""".strip())

Q_TO_A = Template("""
Provide a concise and technically accurate answer to the following question.
Do not be conversational. Output only the answer.

Question: "{{ question }}"
Answer:
""".strip())

JUDGE_SYS = "You are an expert technical evaluator. Your responses MUST be in JSON format."
JUDGE_USER_TMPL = Template("""
You will be given an Original Answer and a Transformed Answer.
Your task is to assess how much of the core technical meaning and information from the
Original Answer is preserved in the Transformed Answer.
Focus on factual accuracy and semantic similarity.

Original Answer:
{{ original }}

Transformed Answer:
{{ transformed }}

Provide your evaluation as a JSON object with a single key "score", where the value is an
integer between 1 (completely unrelated) and 10 (perfectly preserved).
""".strip())
