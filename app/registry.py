"""Algorithm code → detector backend + prompt routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlgorithmSpec:
    code: str
    name: str
    stage: str  # detect | behavior
    backend: str = "world"  # world | fire_smoke | (future: helmet ...)
    prompts: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    description: str = ""
    enabled: bool = True
    # 0.0 = use backend/global conf (no per-algo post-filter)
    conf_threshold: float = 0.0


# Align with recognition_algorithm_type seeds + common extensions
ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "ENGINEERING_VEHICLE_DETECTION": AlgorithmSpec(
        code="ENGINEERING_VEHICLE_DETECTION",
        name="工程车检测",
        stage="detect",
        backend="world",
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
        conf_threshold=0.6,
    ),

    "VEHICLE_DETECTION": AlgorithmSpec(
        code="VEHICLE_DETECTION",
        name="车辆检测",
        stage="detect",
        backend="world",
        prompts=("car", "truck", "bus", "van"),
        description="Common vehicles",
        conf_threshold=0.6,
    ),
    "DET_PERSON": AlgorithmSpec(
        code="DET_PERSON",
        name="人员检测",
        stage="detect",
        backend="world",
        prompts=("person",),
        description="Person detection",
        conf_threshold=0.6,
    ),
    "DET_SHIP": AlgorithmSpec(
        code="DET_SHIP",
        name="船舶检测",
        stage="detect",
        backend="world",
        prompts=("boat", "vessel"),
        description="Ships and boats",
        conf_threshold=0.6,
    ),
    "BEH_FISHING": AlgorithmSpec(
        code="BEH_FISHING",
        name="钓鱼行为",
        stage="behavior",
        backend="world",
        prompts=("person",),
        depends_on=("DET_PERSON",),
        description="YOLO-World person + LLM fishing enhancer",
        conf_threshold=0.6,
    ),
    "BEH_FIGHTING": AlgorithmSpec(
        code="BEH_FIGHTING",
        name="打架行为",
        stage="behavior",
        backend="world",
        prompts=("person",),
        depends_on=("DET_PERSON",),
        description="YOLO-World person + LLM fighting enhancer",
        conf_threshold=0.6,
    ),
    "BEH_SMOKING": AlgorithmSpec(
        code="BEH_SMOKING",
        name="抽烟行为",
        stage="behavior",
        backend="world",
        prompts=("person",),
        depends_on=("DET_PERSON",),
        description="YOLO-World person + LLM smoking enhancer",
        conf_threshold=0.6,
    ),
    "DET_FIRE_SMOKE": AlgorithmSpec(
        code="DET_FIRE_SMOKE",
        name="火焰烟雾检测",
        stage="detect",
        backend="fire_smoke",
        prompts=(),
        description="Dedicated YOLOv8 fire/smoke weights",
        conf_threshold=0.4,
    ),
    "BEH_FIRE_SMOKE": AlgorithmSpec(
        code="BEH_FIRE_SMOKE",
        name="烟火确认",
        stage="behavior",
        backend="fire_smoke",
        prompts=(),
        depends_on=("DET_FIRE_SMOKE",),
        description="Dedicated fire/smoke YOLO + LLM confirmation",
        conf_threshold=0.5,
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
            "backend": s.backend,
            "prompts": list(s.prompts),
            "dependsOn": list(s.depends_on),
            "description": s.description,
            "enabled": s.enabled,
        }
        for s in ALGORITHM_SPECS.values()
        if s.enabled
    ]


def collect_prompts(codes: list[str], *, backend: str | None = "world") -> list[str]:
    """Merge unique prompts.

    When ``backend`` is set (default ``world``), only specs with that backend
    contribute prompts. Pass ``backend=None`` to include all backends.
    """
    prompts: list[str] = []
    seen: set[str] = set()
    for code in codes:
        spec = get_spec(code)
        if spec is None:
            continue
        if backend is not None and spec.backend != backend:
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
    # backend → algorithm codes that use that detector
    by_backend: dict[str, list[str]] = field(default_factory=dict)


def plan_request(algorithm_types: list[str] | None) -> SlotPlan:
    codes = [c.strip() for c in (algorithm_types or []) if c and c.strip()]
    known: list[str] = []
    unknown: list[str] = []
    by_backend: dict[str, list[str]] = {}
    for code in codes:
        spec = get_spec(code)
        if spec is None:
            unknown.append(code)
            continue
        known.append(code)
        by_backend.setdefault(spec.backend, []).append(code)
    return SlotPlan(
        codes=codes,
        known=known,
        unknown=unknown,
        prompts=collect_prompts(known, backend="world"),
        by_backend=by_backend,
    )
