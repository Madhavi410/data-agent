"""
Data Agent
----------
Loads a CSV into pandas and lets Claude answer natural-language questions about it.

How it works (the agent loop):
  1. The model sees the CSV schema + a few sample rows.
  2. For each question, it writes pandas code and calls the `run_pandas` tool.
  3. We execute the code on the loaded data and return the output.
  4. The model can run more code if needed, then gives the final answer.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python data_agent.py sales.csv
"""

import io
import os
import sys
import contextlib
import traceback

import numpy as np
import pandas as pd
import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 8            # max tool calls per question (prevents infinite loops)
MAX_OUTPUT_CHARS = 4000  # truncate big tool outputs


TOOLS = [
    {
        "name": "run_pandas",
        "description": (
            "Execute Python (pandas/numpy) code against the loaded dataset. "
            "The DataFrame is available as `df`; `pd` and `np` are also available. "
            "Use print() to output anything you want to see. "
            "Do not modify files or import modules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Must print() its results.",
                }
            },
            "required": ["code"],
        },
    }
]

# Very small allow-list of builtins for the executed code.
SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "len", "list", "map", "max", "min", "print", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "isinstance", "repr",
    ]
}


class DataAgent:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.history = []                    # keeps conversation memory
        self.system = self._build_system_prompt(csv_path)

    # ---------- prompt ----------
    def _build_system_prompt(self, csv_path: str) -> str:
        buf = io.StringIO()
        self.df.info(buf=buf)
        return f"""You are a data analyst agent. A CSV file ({os.path.basename(csv_path)}) is loaded
as a pandas DataFrame named `df`.

DATASET INFO:
{buf.getvalue()}

FIRST 5 ROWS:
{self.df.head().to_string()}

RULES:
- Answer ONLY questions related to this dataset. If a question is unrelated, politely say
  you can only answer questions about the loaded data.
- Never guess numbers. Always compute answers with the `run_pandas` tool.
- If code errors, read the error, fix the code and retry.
- Check column names/values (e.g. spelling, case) before filtering if a result is empty.
- Give a clear, concise final answer in plain language, including the key numbers."""

    # ---------- tool execution ----------
    def _run_pandas(self, code: str) -> str:
        env = {"__builtins__": SAFE_BUILTINS, "df": self.df.copy(), "pd": pd, "np": np}
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(code, env)  # NOTE: not a hardened sandbox, see README note
            result = out.getvalue().strip() or "(code ran but printed nothing — use print())"
        except Exception:
            result = "ERROR:\n" + traceback.format_exc(limit=2)
        return result[:MAX_OUTPUT_CHARS]

    # ---------- agent loop ----------
    def ask(self, question: str, verbose: bool = True) -> str:
        self.history.append({"role": "user", "content": question})

        for _ in range(MAX_STEPS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=self.system,
                tools=TOOLS,
                messages=self.history,
            )
            self.history.append({"role": "assistant", "content": response.content})

            # Model is done -> return its final text
            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            # Otherwise run each requested tool and feed results back
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    code = block.input.get("code", "")
                    if verbose:
                        print(f"\n[agent code]\n{code}")
                    output = self._run_pandas(code)
                    if verbose:
                        print(f"[output]\n{output}")
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": output}
                    )
            self.history.append({"role": "user", "content": tool_results})

        return "I couldn't reach a final answer within the step limit. Try rephrasing the question."


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_agent.py <path_to_csv>")
        sys.exit(1)

    agent = DataAgent(sys.argv[1])
    print(f"Loaded {len(agent.df)} rows x {len(agent.df.columns)} columns.")
    print("Ask questions about your data (type 'exit' to quit).\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            print(f"\nAgent: {agent.ask(q)}\n")


if __name__ == "__main__":
    main()
