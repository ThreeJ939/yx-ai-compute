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
    spec = get_spec("FISHING_DETECTION")
    assert spec is not None
    assert spec.prompts == ("person",)
    assert spec.stage == "behavior"


def test_collect_prompts_dedupes():
    prompts = collect_prompts(
        ["FISHING_DETECTION", "DET_PERSON", "ENGINEERING_VEHICLE_DETECTION"]
    )
    assert prompts.count("person") == 1
    assert "excavator" in prompts


def test_plan_request_unknown():
    plan = plan_request(["FISHING_DETECTION", "NO_SUCH_ALGO"])
    assert plan.known == ["FISHING_DETECTION"]
    assert plan.unknown == ["NO_SUCH_ALGO"]
    assert "person" in plan.prompts


def test_prompt_matches():
    assert prompt_matches_detection("person", "person", "person")
    assert prompt_matches_detection("dump truck", "dump truck", "dump truck")
    assert prompt_matches_detection("truck", "dump truck", "dump truck")
