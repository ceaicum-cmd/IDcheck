def dist_3d(a, b):
    import numpy as np
    return np.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a.get("z",0)-b.get("z",0))**2)

def get_scale_factor(world_landmarks, real_height_cm=None, face_ipd_mm=None):
    # Multi-reference calibration logic
    pass  # Placeholder - full implementation already in local file