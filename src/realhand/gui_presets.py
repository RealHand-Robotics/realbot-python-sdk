"""Hand preset and initial-position definitions for the RealHand GUI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HandConfig:
    joint_names: list[str] = field(default_factory=list)
    init_pos: list[int] = field(default_factory=list)
    preset_actions: dict[str, list[int]] = field(default_factory=dict)
    cycle_loop_actions: list[str] = field(default_factory=list)
    cycle_loop_repeats: int = 0


def _scale_255(values: list[int]) -> list[int]:
    """Convert old 0-255 example presets to the new SDK's 0-100 range."""
    return [round(max(0, min(255, value)) * 100 / 255) for value in values]


HAND_CONFIGS: dict[str, HandConfig] = {
    "L6": HandConfig(
        joint_names=[
            "Thumb flexion",
            "Thumb yaw",
            "Index finger flexion",
            "Middle finger flexion",
            "Ring finger flexion",
            "Pinky finger flexion",
        ],
        init_pos=_scale_255([250, 250, 250, 250, 250, 250]),
        preset_actions={
            "Open": _scale_255([250, 250, 250, 250, 250, 250]),
            "One": _scale_255([0, 18, 255, 0, 0, 0]),
            "Two": _scale_255([0, 39, 255, 255, 0, 0]),
            "Three": _scale_255([0, 39, 255, 255, 255, 0]),
            "Four": _scale_255([0, 0, 255, 255, 255, 255]),
            "Five": _scale_255([255, 255, 255, 255, 255, 255]),
            "OK": _scale_255([74, 13, 153, 255, 255, 255]),
            "Thumbs Up": _scale_255([255, 255, 0, 0, 0, 0]),
            "Fist": _scale_255([79, 11, 0, 0, 0, 0]),
            "Thumb In": [98, 0, 98, 98, 98, 98],
            "Grab": [33, 0, 76, 68, 66, 58]

        },
    ),
    "O6": HandConfig(
        joint_names=[
            "Thumb flexion",
            "Thumb yaw",
            "Index finger flexion",
            "Middle finger flexion",
            "Ring finger flexion",
            "Pinky finger flexion",
        ],
        init_pos=_scale_255([250, 250, 250, 250, 250, 250]),
        preset_actions={
            "Open": _scale_255([250, 250, 250, 250, 250, 250]),
            "One": _scale_255([125, 18, 255, 0, 0, 0]),
            "Two": _scale_255([92, 87, 255, 255, 0, 0]),
            "Three": _scale_255([92, 87, 255, 255, 255, 0]),
            "Four": _scale_255([92, 87, 255, 255, 255, 255]),
            "Five": _scale_255([255, 255, 255, 255, 255, 255]),
            "OK": _scale_255([96, 100, 118, 250, 250, 250]),
            "Thumbs Up": _scale_255([250, 79, 0, 0, 0, 0]),
            "Fist": _scale_255([102, 18, 0, 0, 0, 0]),
            "Thumb_in" : [100, 33, 98, 98, 98, 98],
            "Two Finger": [100, 17, 100, 100, 0, 0],
 	    "Three Finger": [100, 17, 100, 100, 100, 0]
        },
    ),
    "L20lite": HandConfig(
        joint_names=[
            "Thumb flexion",
            "Thumb abduction",
            "Index finger flexion",
            "Middle finger flexion",
            "Ring finger flexion",
            "Pinky finger flexion",
            "Index finger abduction",
            "Ring finger abduction",
            "Pinky finger abduction",
            "Thumb yaw",
        ],
        init_pos=[100, 100, 100, 100, 100, 100, 50, 50, 50, 100],
        preset_actions={
            "Open": [100, 100, 100, 100, 100, 100, 50, 50, 50, 100],
            "Fist": [15, 35, 0, 0, 0, 0, 50, 50, 50, 30],
            "One": [15, 35, 100, 0, 0, 0, 50, 50, 50, 30],
            "Two": [15, 35, 100, 100, 0, 0, 50, 50, 50, 30],
            "Three": [15, 35, 100, 100, 100, 0, 50, 50, 50, 30],
            "OK": [35, 35, 35, 100, 100, 100, 50, 50, 50, 35],
            "Thumbs Up": [100, 100, 0, 0, 0, 0, 50, 50, 50, 100],
        },
    ),
    "L20": HandConfig(
        joint_names=[
            "Thumb abduction",
            "Thumb yaw",
            "Thumb root",
            "Thumb tip",
            "Index abduction",
            "Index root",
            "Index tip",
            "Middle abduction",
            "Middle root",
            "Middle tip",
            "Ring abduction",
            "Ring root",
            "Ring tip",
            "Pinky abduction",
            "Pinky root",
            "Pinky tip",
        ],
        init_pos=[100, 100, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
        preset_actions={
            "Open": [100, 100, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
            "Fist": [70, 62, 31, 52, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0],
	    "Fist Release": [70, 62, 60, 52, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            "One": [45, 40, 70, 25, 50, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0],
	    "Two": [48, 21, 53, 40, 50, 100, 100, 50, 100, 100, 50, 0, 0, 50, 0, 0],
	    "Three": [38, 12, 53, 40, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 0, 0],
	    
	    
	    "Thumb in": [40, 49, 100, 49, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
	    "Pre-grab": [61, 45, 73, 38, 52, 71, 33, 50, 100, 100, 50, 100, 100, 50, 100, 100],
            "OK": [61, 45, 73, 38, 52, 24, 33, 50, 100, 100, 50, 100, 100, 50, 100, 100],
	    "Thumb to middle": [61, 31, 75, 38, 52, 100, 100, 56, 27, 28, 50, 100, 100, 50, 100, 100],
	    "Thumb to ring": [53, 20, 73, 38, 52, 100, 100, 56, 100, 100, 50, 28, 31, 50, 100, 100],
            "Thumbs Up": [100, 100, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            "All right": [100, 100, 100, 100, 0, 100, 100, 0, 100, 100, 0, 100, 100, 0, 100, 100],
            "All left": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "Spread": [100, 100, 100, 100, 0, 100, 100, 17, 100, 100, 71, 100, 100, 100, 100, 100],
            "Squeeze": [100, 100, 100, 100, 88, 100, 100, 48, 100, 100, 42, 100, 100, 25, 100, 100],
            "Grab wide": [100, 45, 100, 100, 0, 66, 12, 17, 60, 21, 75, 52, 27, 100, 47, 29],
            "Tip bend": [100, 55, 100, 0, 50, 100, 0, 50, 100, 0, 50, 100, 0, 50, 100, 0],
            # "Index Left": [45, 40, 70, 25, 100, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            # "Index forward":[45, 40, 70, 25, 50, 64, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            # "Index right": [45, 40, 70, 25, 0, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            "Thumb straight": [26, 48, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
            "Two_finger_pregrab": [21, 52, 100, 52, 50, 100, 100, 51, 100, 100, 54, 0, 0, 50, 0, 0],
            "Two_finger_grab": [21, 57, 82, 52, 50, 0, 55, 51, 0, 59, 54, 0, 0, 50, 0, 0],
            "Grab":[31, 51, 100, 29, 0, 0, 45, 51, 0, 51, 100, 0, 38, 100, 0, 37],




            
            
        },
        cycle_loop_actions=["Index Left", "Index forward", "Index right"],
        cycle_loop_repeats=2,
    ),
    "L25": HandConfig(
        joint_names=[
            "Thumb abduction",
            "Thumb yaw",
            "Thumb root",
            "Thumb tip",
            "Index abduction",
            "Index root",
            "Index tip",
            "Middle abduction",
            "Middle root",
            "Middle tip",
            "Ring abduction",
            "Ring root",
            "Ring tip",
            "Pinky abduction",
            "Pinky root",
            "Pinky tip",
        ],
        init_pos=[100, 100, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
        preset_actions={
            "Open": [100, 100, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 100, 100],
            "Fist": [45, 35, 10, 10, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            "One": [45, 35, 10, 10, 50, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0],
            "Two": [45, 35, 10, 10, 50, 100, 100, 50, 100, 100, 50, 0, 0, 50, 0, 0],
            "Three": [45, 35, 10, 10, 50, 100, 100, 50, 100, 100, 50, 100, 100, 50, 0, 0],
            "OK": [40, 25, 45, 35, 50, 45, 35, 50, 100, 100, 50, 100, 100, 50, 100, 100],
            "Thumbs Up": [100, 100, 100, 100, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0],
        },
    ),
    # L30 positions are native device values, not normalized percentages.
    # The GUI replaces these placeholders with the live readback on connection
    # and obtains each slider's exact V6/V6.2 limits from the hand.
    "L30": HandConfig(
        joint_names=[
            "Thumb base flexion", "Thumb tip flexion", "Thumb side", "Thumb rotation",
            "Ring side", "Ring tip", "Ring base", "Middle base", "Middle tip",
            "Little base", "Little tip", "Little side", "Middle side", "Index side",
            "Index base", "Index tip", "Wrist pitch",
        ],
        init_pos=[0] * 17,
        # No generic motion presets: their safe native values depend on the
        # exact hand, its installation, and the connected firmware.
        preset_actions={},
    ),
}
