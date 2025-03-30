from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class InferenceResult:
    type: Literal["mel", "wave"]
    data: Any
