"""Algorithm code → YOLO-World prompt routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlgorithmSpec:
    code: str
    name: str
    stage: str  # detect | behavior
    prompts: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    description: str = ""
    enabled: bool = True


# Align with recognition_algorithm_type seeds + common extensions
ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "ENGINEERING_VEHICLE_DETECTION": AlgorithmSpec(
        code="ENGINEERING_VEHICLE_DETECTION",
        name="工程车检测",
        stage="detect",
        prompts=(
            "excavator",
            "bulldozer",
            "loader",
            "crane",
            "dump truck",
            "truck",
            "roller",
        ),
        description="YOLO-World open-vocab engineering vehicles",
    ),
    "FISHING_DETECTION": AlgorithmSpec(
        code="FISHING_DETECTION",
        name="钓鱼检测",
        stage="behavior",
        prompts=("person",),
        depends_on=(),
        description="Phase-1: person count via YOLO-World; LLM enhancer is noop",
    ),
    "VEHICLE_DETECTION": AlgorithmSpec(
        code="VEHICLE_DETECTION",
        name="车辆检测",
        stage="detect",
        prompts=("car", "truck", "bus", "van"),
        description="Common vehicles",
    ),
    "DET_PERSON": AlgorithmSpec(
        code="DET_PERSON",
        name="人员检测",
        stage="detect",
        prompts=("person",),
        description="Person detection",
    ),
    "DET_SHIP": AlgorithmSpec(
        code="DET_SHIP",
        name="船舶检测",
        stage="detect",
        prompts=("ship", "boat", "vessel"),
        description="Ships and boats",
    ),
    "BEH_FISHING": AlgorithmSpec(
        code="BEH_FISHING",
        name="钓鱼行为",
        stage="behavior",
        prompts=("person",),
        depends_on=("DET_PERSON",),
        description="Reserved for LLM enhancer (noop in phase-1)",
    ),
    "BEH_FIGHTING": AlgorithmSpec(
        code="BEH_FIGHTING",
        name="打架行为",
        stage="behavior",
        prompts=("person",),
        depends_on=("DET_PERSON",),
        description="Reserved for LLM enhancer (noop in phase-1)",
    ),
}


def get_spec(code: str) -> AlgorithmSpec | None:
    if not code:
        return None
    return ALGORITHM_SPECS.get(code.strip())


def list_capabilities() -> list[dict]:
    return [
        {
            "code": s.code,
            "name": s.name,
            "stage": s.stage,
            "prompts": list(s.prompts),
            "dependsOn": list(s.depends_on),
            "description": s.description,
            "enabled": s.enabled,
        }
        for s in ALGORITHM_SPECS.values()
        if s.enabled
    ]


def collect_prompts(codes: list[str]) -> list[str]:
    """Merge unique prompts for a one-shot YOLO-World forward pass."""
    prompts: list[str] = []
    seen: set[str] = set()
    for code in codes:
        spec = get_spec(code)
        if spec is None:
            continue
        for p in spec.prompts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(p)
    return prompts


def prompt_matches_detection(prompt: str, object_type: str, label: str) -> bool:
    """Loose match between configured prompt and detector class name."""
    p = prompt.strip().lower()
    candidates = {object_type.strip().lower(), label.strip().lower()}
    if p in candidates:
        return True
    # multi-word prompt: allow substring either way
    for c in candidates:
        if p in c or c in p:
            return True
    return False


@dataclass
class SlotPlan:
    codes: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)


def plan_request(algorithm_types: list[str] | None) -> SlotPlan:
    codes = [c.strip() for c in (algorithm_types or []) if c and c.strip()]
    known: list[str] = []
    unknown: list[str] = []
    for code in codes:
        if get_spec(code) is None:
            unknown.append(code)
        else:
            known.append(code)
    return SlotPlan(
        codes=codes,
        known=known,
        unknown=unknown,
        prompts=collect_prompts(known),
    )
