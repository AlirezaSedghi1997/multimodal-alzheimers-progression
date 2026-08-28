"""RED test for Method Y writer blocking - P0."""
import pytest


def test_writer_guard_blocks_method_y_image_training():
    from src.research.brainiac_readiness import _WRITER_MARKERS, _WRITER_BASENAME_MARKERS, _command_is_project_writer

    # RED: Before fix, Method Y markers were missing
    # After fix, they should be present
    assert any("method_y" in m.lower() for m in _WRITER_MARKERS), f"Missing Method Y in _WRITER_MARKERS: {_WRITER_MARKERS}"
    assert any("method_y" in m.lower() for m in _WRITER_BASENAME_MARKERS), f"Missing Method Y in _WRITER_BASENAME_MARKERS: {_WRITER_BASENAME_MARKERS}"
    
    # Test that Method Y command is detected as writer
    method_y_cmd = "python.exe scripts/run_method_y_image_training_v3.py --config test --resume"
    assert _command_is_project_writer(method_y_cmd), "Method Y command should be detected as writer"
    
    # Test that run_method_y_training_v2 is also detected
    method_y_v2_cmd = "python.exe scripts/run_method_y_training_v2.py --config test"
    assert _command_is_project_writer(method_y_v2_cmd), "Method Y v2 command should be detected as writer"
