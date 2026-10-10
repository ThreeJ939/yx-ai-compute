from app.registry import (
    collect_prompts,
    get_spec,
    plan_request,
    prompt_matches_detection,
)


def test_known_engineering_vehicle_spec():
    spec = get_spec("ENGINEERING_VEHICLE_DETECTION")
    assert spec is not None
    assert "excavator" in spec.prompts
    assert spec.stage == "detect"


def test_fishing_uses_person_prompt():
    spec = get_spec("BEH_FISHING")
    assert spec is not None
    assert spec.prompts == ("person",)
    assert spec.stage == "behavior"


def test_collect_prompts_dedupes():
    prompts = collect_prompts(
        ["BEH_FISHING", "DET_PERSON", "ENGINEERING_VEHICLE_DETECTION"]
    )
    assert prompts.count("person") == 1
    assert "excavator" in prompts


def test_plan_request_unknown():
    plan = plan_request(["BEH_FISHING", "NO_SUCH_ALGO"])
    assert plan.known == ["BEH_FISHING"]
    assert plan.unknown == ["NO_SUCH_ALGO"]
    assert "person" in plan.prompts
    assert plan.by_backend.get("world") == ["BEH_FISHING"]


def test_prompt_matches():
    assert prompt_matches_detection("person", "person", "person")
    assert prompt_matches_detection("dump truck", "dump truck", "dump truck")
    assert prompt_matches_detection("truck", "dump truck", "dump truck")


def test_world_backend_default():
    assert get_spec("DET_PERSON").backend == "world"
    assert get_spec("BEH_FISHING").backend == "world"


def test_smoking_behavior_spec():
    spec = get_spec("BEH_SMOKING")
    assert spec is not None
    assert spec.stage == "behavior"
    assert spec.backend == "world"
    assert spec.prompts == ("person",)


def test_algo_conf_threshold_default_zero():
    """Per-algo post-filter is opt-in (0.0 = use backend/global conf only)."""
    for code in (
        "DET_PERSON",
        "DET_SHIP",
        "BEH_FISHING",
        "DET_FIRE_SMOKE",
        "BEH_FIRE_SMOKE",
    ):
        spec = get_spec(code)
        assert spec is not None
        assert spec.conf_threshold == 0.0
