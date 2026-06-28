from __future__ import annotations

from typing import Any, Mapping

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

IDENTITY_3: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

ROT_Z_POS_90: Matrix3 = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)

ROT_Z_NEG_90: Matrix3 = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)

ROT_Z_180: Matrix3 = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)

ROT_Y_180: Matrix3 = (
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
)

ORIENTATION_PRESETS: dict[str, Matrix3] = {
    "identity": IDENTITY_3,
    "rot_z_pos_90": ROT_Z_POS_90,
    "rot_z_neg_90": ROT_Z_NEG_90,
    "rot_z_180": ROT_Z_180,
    "rot_y_180": ROT_Y_180,
}

MAG_LAYOUTS: dict[str, dict[str, str]] = {
    "rear_v1": {
        "mag": "identity",
        "mag_lis": "identity",
    },
    "rear_legacy_lis_z_neg_90": {
        "mag": "identity",
        "mag_lis": "rot_z_neg_90",
    },
    "rear_v2_lis_y180": {
        "mag": "identity",
        "mag_lis": "rot_y_180",
    },
}


def matrix_for_preset(name: str) -> list[list[float]]:
    try:
        matrix = ORIENTATION_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ORIENTATION_PRESETS))
        raise ValueError(f"Unknown orientation preset '{name}'. Choices: {choices}") from exc
    return [list(row) for row in matrix]


def signal_configs_for_mag_layout(layout_name: str) -> dict[str, dict[str, Any]]:
    try:
        layout = MAG_LAYOUTS[layout_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MAG_LAYOUTS))
        raise ValueError(f"Unknown magnetometer layout '{layout_name}'. Choices: {choices}") from exc

    return {
        signal_name: {
            "orientation_preset": preset_name,
            "sensor_to_pod_matrix": matrix_for_preset(preset_name),
        }
        for signal_name, preset_name in layout.items()
    }


def resolve_signal_orientation(signal_config: Mapping[str, Any]) -> list[list[float]] | None:
    matrix = signal_config.get("sensor_to_pod_matrix")
    if matrix is not None:
        return matrix

    preset = signal_config.get("orientation_preset")
    if preset is None:
        return None
    if not isinstance(preset, str):
        raise ValueError(f"orientation_preset must be a string, got {type(preset).__name__}")

    return matrix_for_preset(preset)
