from __future__ import annotations

import json
import logging
import sys
from typing import Any

from brain.client import chat, MODEL_HEAVY
from brain.agents.types import Task

logger = logging.getLogger(__name__)

VALID_TYPES = {"research", "code", "audit", "synthesize", "chat", "tool"}
MAX_TASKS = 12

_SYSTEM_PROMPT = f"""\
You are a task planner for an AI agent system.
Decompose the user request into atomic tasks.

RESPOND WITH ONLY a valid JSON array — no markdown, no text outside JSON.

Each task object has exactly these fields:
  {{
    "id":         "t1",
    "goal":       "<what to do, concise>",
    "type":       "<research|code|audit|synthesize|chat|tool>",
    "tool_name":  null,
    "depends_on": [],
    "inputs":     {{}}
  }}

Maximum {MAX_TASKS} tasks. Every task must be atomic (one clear action).

──────────────────────────────────────────
DEPENDENCY RULES  (CRITICAL — read carefully)
──────────────────────────────────────────
1. depends_on = []  for tasks that do NOT need results from another task.
   EXAMPLE: writing bubble_sort and writing binary_search are INDEPENDENT —
   neither needs the other's code to be written.
   Both get depends_on: []

2. depends_on = ["tX"]  ONLY when this task literally cannot run without tX's output.
   EXAMPLE: an audit task needs the code it audits → depends_on: ["t1"]

3. NEVER chain code tasks together just to "share context".
   Independent functions / modules / components are ALWAYS depends_on: []
   The synthesize task will merge everything at the end.

4. PARALLEL PATTERN — use this when request asks for multiple independent things:
   t1 [code]      func A          depends_on: []
   t2 [code]      func B          depends_on: []
   t3 [code]      func C          depends_on: []
   t4 [audit]     audit A         depends_on: ["t1"]
   t5 [audit]     audit B         depends_on: ["t2"]
   t6 [audit]     audit C         depends_on: ["t3"]
   t7 [synthesize] merge all      depends_on: ["t4", "t5", "t6"]

5. SEQUENTIAL PATTERN — use ONLY when task truly builds on previous output:
   t1 [research]  gather info     depends_on: []
   t2 [code]      use research    depends_on: ["t1"]
   t3 [audit]     audit code      depends_on: ["t2"]
   t4 [synthesize] save result    depends_on: ["t3"]

──────────────────────────────────────────
TASK TYPES
──────────────────────────────────────────
- research   : gather information / investigate a topic
- code       : write or modify code (produces a complete code artifact)
- audit      : review and test code from a prior task (always depends_on a code task)
- synthesize : merge ALL code artifacts, save to disk, tell user path + run command.
               USE ONLY when at least one task has type="code".
               Must depend on ALL final audit/code tasks.
               DO NOT use synthesize if there are NO code tasks in the plan.
- chat       : conversational answer only — use for plans with NO code tasks.
               Final task must be "chat" (not "synthesize") when no code is produced.
- tool       : call a real-time tool to get live data (weather, crypto, currency, time, timer).
               Set "tool_name" to one of the available tools below.
               Set "inputs" to the tool's arguments.
               tool tasks are usually independent (depends_on: []).
               A synthesize or chat task CAN depends_on a tool task to use its result.

──────────────────────────────────────────
AVAILABLE TOOLS  (use only for type="tool")
──────────────────────────────────────────
Use type="tool" when the request needs LIVE data: weather, exchange rates, crypto prices, current time, timers.

  weather(location, language="ru")
    → current weather at a location
    inputs: {{"location": "Москва", "language": "ru"}}

  crypto.search(query)
    → search cryptocurrency by name or symbol
    inputs: {{"query": "bitcoin"}}

  crypto.price(ids[], vs_currency="usd")
    → price(s) for one or more coins by CoinGecko id
    inputs: {{"ids": ["bitcoin", "ethereum"], "vs_currency": "usd"}}

  currency.rates()
    → all current exchange rates (base USD)
    inputs: {{}}

  currency.convert(amount, from_code, to_code)
    → convert amount between currencies
    inputs: {{"amount": 100, "from_code": "USD", "to_code": "RUB"}}

  time()
    → current date and time
    inputs: {{}}

  timer.set(seconds, label)
    → set a countdown timer
    inputs: {{"seconds": 300, "label": "tea"}}

  timer.list()
    → list all active timers
    inputs: {{}}

  timer.cancel(timer_id)
    → cancel a timer by id
    inputs: {{"timer_id": "abc123"}}

  auditor.run(files=[])
    → run code auditor on given file paths
    inputs: {{"files": ["output/script.py"]}}

TOOL TASK EXAMPLES:
  {{"id":"t1","type":"tool","goal":"get current weather in Moscow",
    "tool_name":"weather","inputs":{{"location":"Москва","language":"ru"}},"depends_on":[]}}

  {{"id":"t2","type":"tool","goal":"get USD to RUB exchange rate",
    "tool_name":"currency.convert","inputs":{{"amount":1,"from_code":"USD","to_code":"RUB"}},"depends_on":[]}}

  {{"id":"t3","type":"chat","goal":"write a report based on weather and currency data",
    "tool_name":null,"inputs":{{}},"depends_on":["t1","t2"]}}

──────────────────────────────────────────
FINAL CHECKS before outputting
──────────────────────────────────────────
- If ANY code task exists: last task MUST be "synthesize", depending on ALL final audit/code tasks.
- If NO code tasks exist: last task MUST be "chat" (never "synthesize").
- synthesize depends_on ALL the last audit/code tasks (not just one).
- Independent code tasks have depends_on: [].
- No two code tasks depend on each other unless the second truly extends the first.
- tool tasks always have "tool_name" set to a valid tool name.
- chat/synthesize tasks always have "tool_name": null.

OUTPUT ONLY the JSON array, starting with [ and ending with ].
"""


class PlannerAgent:
    def __init__(self, model: str = MODEL_HEAVY) -> None:
        self.model = model

    def build_prompt(
        self, user_request: str, history: list[dict] | None = None
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if history:
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": f"User request: {user_request}"})
        return messages

    def _call_llm(self, messages: list[dict]) -> str:
        logger.debug("[PlannerAgent] Calling LLM model=%s", self.model)
        return chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.1, "num_ctx": 8192},
        )

    @staticmethod
    def _extract_json(raw: str) -> str:
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
            # FIX: required fields without 'inputs' — inputs is optional
            for field in ("id", "goal", "type", "depends_on"):
                if field not in raw:
                    raise ValueError(f"Task #{i} missing field '{field}'")
            t_id: str = str(raw["id"])
            t_type: str = raw["type"]
            if t_type not in VALID_TYPES:
                raise ValueError(f"Task {t_id} has unknown type '{t_type}'")
            if t_id in ids_seen:
                raise ValueError(f"Duplicate task id '{t_id}'")
            ids_seen.add(t_id)

            if t_type == "tool" and not raw.get("tool_name"):
                raise ValueError(f"Task {t_id} has type='tool' but missing 'tool_name'")

            deps = raw.get("depends_on", [])
            for dep in deps:
                if dep not in ids_seen:
                    raise ValueError(
                        f"Task {t_id} depends on '{dep}' which is not yet defined — "
                        "tasks must be in dependency order"
                    )

            # FIX: inputs is optional — default to empty dict
            raw_inputs = raw.get("inputs", {}) or {}
            raw["inputs"] = raw_inputs if isinstance(raw_inputs, dict) else {}

            result.append(Task.from_dict(raw))

        has_code = any(t.type == "code" for t in result)

        # FIX BUG-B: if LLM returned synthesize as last task but there are NO code tasks,
        # downgrade it to "chat" type to avoid saving Python files for non-code plans
        if not has_code and result and result[-1].type == "synthesize":
            logger.warning(
                "[PlannerAgent] Last task is 'synthesize' but no code tasks found — "
                "downgrading to 'chat'"
            )
            last = result[-1]
            result[-1] = Task(
                id=last.id,
                goal=last.goal,
                type="chat",
                tool_name=None,
                depends_on=last.depends_on,
                inputs=last.inputs,
            )

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

    def plan(
        self, user_request: str, history: list[dict] | None = None
    ) -> list[Task]:
        messages = self.build_prompt(user_request, history=history)
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
        tool = f" [tool={task.tool_name}]" if task.tool_name else ""
        print(f"  [{task.id}] [{task.type.upper()}]{tool} {task.goal}{deps}")
        if task.inputs:
            print(f"         inputs: {task.inputs}")
    print()
