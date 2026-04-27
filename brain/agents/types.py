from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Task:
    id: str                    # "t1", "t2", ...
    goal: str                  # что сделать
    type: Literal["research", "code", "audit", "synthesize", "chat"]
    depends_on: list[str] = field(default_factory=list)  # ["t1"]
    inputs: dict = field(default_factory=dict)
    status: str = "pending"    # pending | running | done | failed
    artifact: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "type": self.type,
            "depends_on": self.depends_on,
            "inputs": self.inputs,
            "status": self.status,
            "artifact": self.artifact,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            goal=data["goal"],
            type=data["type"],
            depends_on=data.get("depends_on", []),
            inputs=data.get("inputs", {}),
            status=data.get("status", "pending"),
            artifact=data.get("artifact"),
        )
