from __future__ import annotations

from pathlib import Path

from echochain.records import make_run_dir, prepare_resume, read_manifest, slug_for


def test_slug_stays_off_the_directory_name(tmp_path: Path) -> None:
    seed = "Redis epoll event"
    allocated = make_run_dir(seed, tmp_path)
    assert slug_for(seed) == "redis_epoll_event"
    assert "redis" not in allocated.directory.name
    assert allocated.slug == "redis_epoll_event"


def test_resume_from_a_manifest_without_rows(tmp_path: Path) -> None:
    from echochain.records import (
        BaselinePins,
        ErrorCounts,
        Manifest,
        RunStatus,
        Schedule,
        StopName,
        StopRule,
        Temperatures,
        sha256_text,
        write_manifest,
    )

    allocated = make_run_dir("stored seed", tmp_path)
    text = "stored seed"
    manifest = Manifest(
        status=RunStatus.failed,
        run_id=allocated.run_id,
        slug=allocated.slug,
        version="0.1.0",
        python="3.11.0",
        os="test",
        seed_sha256=sha256_text(text),
        seed_text=text,
        seed_id="custom",
        claims=[],
        condition="alternate",
        schedule=Schedule(name="alternate", models=["vendor-a/model"]),
        requested_models=["vendor-a/model"],
        served_models=[],
        providers=[],
        temperatures=Temperatures(transform=0.3, judge=0.5),
        top_p=1,
        sampling_seed=4,
        max_cycles=4,
        max_tokens=32,
        stop_rule=StopRule(name=StopName.fixed_length, threshold=2, consecutive=0),
        max_judge_error_share=0.1,
        judge_panel=[],
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        prompt_tokens=0,
        completion_tokens=0,
        cost=0,
        errors=ErrorCounts(generation=0, judge=0),
        termination_reason=None,
        budget=None,
        deadline_seconds=60,
        error=None,
        batch_id=None,
        repeat_index=0,
        baseline=BaselinePins(),
    )
    write_manifest(allocated.directory / "manifest.json", manifest)
    state = prepare_resume(allocated.directory)
    assert state.next_step == 1
    assert state.seed == text
    assert read_manifest(allocated.directory).seed_text == text
