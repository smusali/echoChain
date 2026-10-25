from jinja2 import Template

RUBRIC_VERSION = "1"

A_TO_Q_SOURCE = """
Given the following statement, formulate a concise technical question for which this
statement is the direct and most appropriate answer.
Do not be conversational. Output only the question.

Statement: "{{ answer }}"
Question:
""".strip()

Q_TO_A_SOURCE = """
Provide a concise and technically accurate answer to the following question.
Do not be conversational. Output only the answer.

Question: "{{ question }}"
Answer:
""".strip()

IDENTITY_SOURCE = """
Repeat the following text verbatim.
Do not be conversational. Output only the text.

Text:
{{ text }}
""".strip()

PARAPHRASE_SOURCE = """
Paraphrase the following technical statement.
Preserve the technical meaning.
Do not be conversational. Output only the paraphrase.

Statement:
{{ text }}
""".strip()

JUDGE_USER_SOURCE = """
You will be given an Original Answer and a Transformed Answer.
Your task is to assess how much of the core technical meaning and information from the
Original Answer is preserved in the Transformed Answer.
Focus on factual accuracy and semantic similarity.

Original Answer:
{{ original }}

Transformed Answer:
{{ transformed }}

Provide your evaluation as a JSON object with the keys "rationale" and "score", in that
order. "rationale" is a short string. "score" is an integer between 1 (completely
unrelated) and 10 (perfectly preserved).
Example:
{"rationale": "The mechanism is preserved.", "score": 8}
""".strip()

ANSWERABILITY_USER_SOURCE = """
You will be given a Question and a Seed Answer.
Decide whether the Seed Answer would be a complete, correct answer to the Question.
Focus on whether the question asks for the seed's technical claim.

Question:
{{ question }}

Seed Answer:
{{ seed }}

Provide your evaluation as a JSON object with the keys "rationale" and "score", in that
order. "rationale" is a short string. "score" is an integer between 1 (the seed would not
answer the question) and 10 (the seed would answer it completely).
Example:
{"rationale": "The seed answers the question.", "score": 8}
""".strip()

A_TO_Q = Template(A_TO_Q_SOURCE)
Q_TO_A = Template(Q_TO_A_SOURCE)
IDENTITY_TMPL = Template(IDENTITY_SOURCE)
PARAPHRASE_TMPL = Template(PARAPHRASE_SOURCE)
JUDGE_SYS = (
    "You are an expert technical evaluator. Your responses MUST be in JSON format."
)
JUDGE_USER_TMPL = Template(JUDGE_USER_SOURCE)
ANSWERABILITY_USER_TMPL = Template(ANSWERABILITY_USER_SOURCE)
