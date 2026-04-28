"""brain/agents/plan_agent.py

Точка входа для route="plan".
Запускает PlannerAgent → Executor → возвращает финальный артефакт.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run(query: str, history: list[dict] | None = None) -> str:
    from brain.agents.planner import PlannerAgent
    from brain.agents.executor import Executor

    logger.info("[plan_agent] Planning query: %s", query[:120])

    planner = PlannerAgent()
    try:
        # FIX-1: пробрасываем history в planner, чтобы контекст предыдущего
        # аудита/диалога был доступен при построении плана.
        tasks = planner.plan(query, history=history)
    except Exception as e:
        logger.error("[plan_agent] Planner failed: %s", e)
        return f"Планировщик не смог разбить задачу: {e}"

    plan_summary = "\n".join(
        f"  [{t.id}] [{t.type.upper()}] {t.goal}"
        + (f" ← {t.depends_on}" if t.depends_on else "")
        for t in tasks
    )
    logger.info("[plan_agent] Plan (%d tasks):\n%s", len(tasks), plan_summary)

    executor = Executor(critic_retries=3, use_parallel=True)
    try:
        context = executor.run(tasks, user_request=query)
    except Exception as e:
        logger.error("[plan_agent] Executor failed: %s", e)
        return f"Исполнитель упал с ошибкой: {e}"

    done_tasks   = [t for t in tasks if t.status == "done"]
    failed_tasks = [t for t in tasks if t.status == "failed"]

    # FIX-4: предпочитаем synthesize-таск как финальный артефакт;
    # если его нет среди выполненных — берём последний done-таск.
    synth_task = next(
        (t for t in reversed(done_tasks) if t.type == "synthesize"), None
    )
    final_task = synth_task or (done_tasks[-1] if done_tasks else None)
    final_artifact = final_task.artifact if final_task else ""

    header = (
        f"✅ Выполнено {len(done_tasks)}/{len(tasks)} задач"
        + (f"  ❌ Не удалось: {[t.id for t in failed_tasks]}" if failed_tasks else "")
        + "\n\n"
    )

    return header + final_artifact if final_artifact else header + "Задачи выполнены, но финального артефакта нет."
