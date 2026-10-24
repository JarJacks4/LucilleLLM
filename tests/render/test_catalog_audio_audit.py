"""Soundscape bucket inventory is read-only and uses catalog-defined asset paths."""

from render_service.catalog_audit import missing_by_track, object_paths, render_commands


def _track(track_id="jazz_01"):
    return {
        "id": track_id,
        "audio": {
            "path": f"soundscapes/jazz/{track_id}/",
            "segments": {
                "intro": "intro.m4a",
                "bodies": ["body_1.m4a", "body_2.m4a", "body_3.m4a", "body_4.m4a"],
                "outro": "outro.m4a",
            },
            "preview": "preview.m4a",
        },
    }


def test_catalog_audio_expected_gcs_keys():
    names = object_paths(_track())
    assert len(names) == 7
    assert names[0] == "soundscapes/jazz/jazz_01/intro.m4a"
    assert names[-1] == "soundscapes/jazz/jazz_01/preview.m4a"
    assert names[5].endswith("outro.m4a")


def test_missing_inventory_identifies_incomplete_tracks():
    a, b = _track("jazz_01"), _track("jazz_02")
    complete_a = object_paths(a)
    complete_b = object_paths(b)
    missing = missing_by_track([a, b], complete_a + complete_b[1:])
    assert missing == {"jazz_02": [complete_b[0]]}
    assert missing_by_track([a, b], complete_a + complete_b) == {}


def test_render_commands_are_batched_and_nonexecuting():
    commands = render_commands(["jazz_01", "jazz_02", "jazz_03"], "my-bucket", batch_size=2)
    assert len(commands) == 2
    assert commands[0] == "AUDIO_BUCKET=my-bucket python -m render_service.catalog --only jazz_01 jazz_02"
    assert commands[1].endswith("--only jazz_03")
