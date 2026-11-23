import json
import re

from langchain_groq import ChatGroq

from config import GROQ_KEY, LLM_MODELS


def _ask(model, prompt):
    llm = ChatGroq(model=model, api_key=GROQ_KEY, temperature=0.3, max_retries=0, timeout=30)
    return llm.invoke(prompt).content


def _ask_json(model, prompt, keys):
    raw = _ask(model, prompt).strip()
    clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    data = json.loads(clean)
    if not all(isinstance(data.get(k), str) and data[k] for k in keys):
        raise ValueError("response missing required keys")
    return clean


def generate(log, agent, prompt, template, keys=None):
    for model in LLM_MODELS:
        if keys:
            fn = lambda m=model: _ask_json(m, prompt, keys)
        else:
            fn = lambda m=model: _ask(m, prompt)
        out = log.call(agent, model, prompt, fn)
        if out:
            return (json.loads(out) if keys else out), model
    log.call(agent, "template", prompt, lambda: json.dumps(template) if keys else template)
    return template, "template"
