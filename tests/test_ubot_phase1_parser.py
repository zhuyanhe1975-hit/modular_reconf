import pytest
from pathlib import Path
import os
from ubot.spec import FaceID
from ubot.mjcf_parser import load_ubot_mjcf

UBOT_MJCF_PATH = os.getenv("UBOT_MJCF_PATH")

def create_dummy_mjcf():
    """Create a dummy MJCF string for testing."""
    return """<mujoco>
<worldbody>
<body name="ax_center">
</body>
<body name="ma_half">
<joint type="hinge" name="j1" axis="0 0 1" range="-3.14159 3.14159" pos="0 0 0"/>
<geom type="box" name="box_ma"/>
</body>
<body name="mb_half">
<joint type="hinge" name="j2" axis="0 0 1" range="-3.14159 3.14159" pos="0 0 0"/>
<geom type="box" name="box_mb"/>
</body>
</worldbody>
</mujoco>"""

@pytest.mark.skipif(not UBOT_MJCF_PATH, reason="UBOT_MJCF_PATH not set")
def test_load_ubot_mjcf():
    """Test parsing real MJCF."""
    spec = load_ubot_mjcf(UBOT_MJCF_PATH)
    assert spec is not None
    assert len(spec.joints) == 2  # 2 internal joints
    assert len(spec.halves["ma"].faces) == 2  # 2 faces
    assert len(spec.halves["mb"].faces) == 2  # 2 faces
    all_faces = list(spec.halves["ma"].faces.keys()) + list(spec.halves["mb"].faces.keys())
    assert FaceID.MA_RIGHT in spec.halves["ma"].faces
    assert FaceID.MB_LEFT in spec.halves["mb"].faces
    # More assertions as needed

def test_load_dummy_mjcf():
    """Test parsing dummy MJCF string."""
    # Save dummy to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(create_dummy_mjcf())
        temp_path = f.name
    try:
        spec = load_ubot_mjcf(temp_path)
        assert spec.ax_body_name == "ax_center"
        assert spec.ma_body_name == "ma_half"
        assert spec.mb_body_name == "mb_half"
        assert len(spec.joints) == 2
    finally:
        Path(temp_path).unlink()
