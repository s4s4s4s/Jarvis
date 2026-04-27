from __future__ import annotations

import json
import logging
import sys
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.agents.types import Task

logger = logging.getLogger(__name__)

VALID_TYPES = {"research", "code", "audit", "synthesize", "chat"}
MAX_TASKS = 12

_SYSTEM_PROMPT = f"""\
You are a task planner for an AI agent system. Your job is to decompose a user request
into an ordered list of atomic tasks that will be executed sequentially.

RULES:
- Respond with ONLY a valid JSON array. No markdown, no explanations, no text outside JSON.
- Each task object must have exactly these fields:
  {{
    "id": "t1",
    "goal": "<what to do, concise>",
    "type": "<research|code|audit|synthesize|chat>",
    "depends_on": ["t1"],
    "inputs": {{}}
  }}
- Maximum {MAX_TASKS} tasks.
- Every task must be atomic (one clear action).
- depends_on must list ALL task ids whose artifacts this task needs as input.
  Example: if t5 needs results from t3 AND t4, use "depends_on": ["t3", "t4"].
- If no dependencies, use empty list [].

TASK TYPES:
- research   : gather information, read docs, investigate a topic
- code       : write or modify code (always produces a complete code artifact)
- audit      : review and test code from a previous task (always depends_on a code task)
- synthesize : MANDATORY last task — combine ALL code artifacts into one final deliverable,
               save to disk, and tell the user where the file is and how to run it.
               Must depend on the last audit or code task.
- chat       : ONLY for non-code requests that need a conversational answer.
               Do NOT use chat as the final task when code is being produced.

CRITICAL RULES:
1. When the request involves writing code, the LAST task MUST be type "synthesize".
   NEVER end a code pipeline with "chat".
2. The synthesize task must have depends_on pointing to the last code/audit task
   so it receives the final code artifact.
3. Every code task should depend on the previous code or audit task to accumulate
   all prior code context — do NOT start fresh each time.
4. After every code task, add an audit task to verify it before proceeding.

IMPORTANT: Output ONLY the JSON array, starting with [ and ending with ].
"""


class PlannerAgent:
    def __init__(self, model: str = MODEL_HEAVY) -> None:
        self.model = model

    def build_prompt(self, user_request: str) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"User request: {user_request}"},
        ]

    def _call_llm(self, messages: list[dict]) -> str:
        logger.debug("[PlannerAgent] Calling LLM model=%s", self.model)
        return chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.1, "num_ctx": 8192},
        )

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Strip any accidental markdown fences or leading/trailing text."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            raw = "\n".join(inner).strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON array found in LLM output: {raw[:200]}")
        return raw[start : end + 1]

    @staticmethod
    def _validate(tasks: list[dict]) -> list[Task]:
        if not tasks:
            raise ValueError("Plan is empty")
        if len(tasks) > MAX_TASKS:
            raise ValueError(f"Too many tasks: {len(tasks)} > {MAX_TASKS}")

        ids_seen: set[str] = set()
        result: list[Task] = []
        for i, raw in enumerate(tasks):
            for field in ("id", "goal", "type", "depends_on", "inputs"):
                if field not in raw:
                    raise ValueError(f"Task #{i} missing field '{field}'")
            t_id: str = str(raw["id"])
            t_type: str = raw["type"]
            if t_type not in VALID_TYPES:
                raise ValueError(f"Task {t_id} has unknown type '{t_type}'")
            if t_id in ids_seen:
                raise ValueError(f"Duplicate task id '{t_id}'")
            ids_seen.add(t_id)

            deps = raw.get("depends_on", [])
            for dep in deps:
                if dep not in ids_seen:
                    raise ValueError(
                        f"Task {t_id} depends on '{dep}' which is not yet defined — "
                        "tasks must be in dependency order"
                    )
            result.append(Task.from_dict(raw))

        # Post-validation: if any code tasks exist, last task must be synthesize
        has_code = any(t.type == "code" for t in result)
        if has_code and result[-1].type != "synthesize":
            logger.warning(
                "[PlannerAgent] Last task is '%s' but code was produced — "
                "auto-appending synthesize task",
                result[-1].type,
            )
            last_id = result[-1].id
            synth = Task(
                id=f"t{len(result)+1}",
                goal="Combine all code artifacts into one final script, save to disk, and report the file path and usage instructions to the user.",
                type="synthesize",
                depends_on=[last_id],
                inputs={},
            )
            result.append(synth)

        return result

    def plan(self, user_request: str) -> list[Task]:
        messages = self.build_prompt(user_request)
        raw_output = self._call_llm(messages)
        logger.debug("[PlannerAgent] Raw LLM output: %s", raw_output[:500])

        try:
            json_str = self._extract_json(raw_output)
            data: list[Any] = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("[PlannerAgent] JSON parse failed: %s", e)
            raise

        tasks = self._validate(data)
        logger.info("[PlannerAgent] Plan created: %d tasks", len(tasks))
        return tasks


# ----------------------------------------------------------------------
# CLI: python -m brain.agents.planner "<user request>"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print('Usage: python -m brain.agents.planner "<user request>"')
        sys.exit(1)

    user_req = " ".join(sys.argv[1:])
    print(f"\n[Planner] Request: {user_req}\n")

    agent = PlannerAgent()
    try:
        plan = agent.plan(user_req)
    except Exception as exc:
        print(f"[Planner] ERROR: {exc}")
        sys.exit(1)

    print(f"[Planner] Generated {len(plan)} tasks:\n")
    for task in plan:
        deps = f" (depends: {task.depends_on})" if task.depends_on else ""
        print(f"  [{task.id}] [{task.type.upper()}] {task.goal}{deps}")
        if task.inputs:
            print(f"         inputs: {task.inputs}")
    print()
