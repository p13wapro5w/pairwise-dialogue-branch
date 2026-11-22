import pytest
import gameplay.dialogue_branch as mod

CASES = [
    ("restore_options", (1, 2)),
    ("server_authority", (1, 2)),
    ("seen_marker", (1, 2)),
    ("mid_save", (1, 2)),
    ("voice_wait", (2.0, 1.0)),
    ("layout_stable", (1, 2)),
    ("speaker_lock", (1, 2)),
    ("timeline_unify", (1, 2)),
    ("weighted_random", (42, 3)),
    ("skip_compensate", (1, 2)),
    ("condition_eval", (1, 2)),
    ("option_order", (3, 2)),
    ("history_stack", (1, 2)),
    ("auto_play", (1, 2)),
    ("localization_width", (20, 40)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
