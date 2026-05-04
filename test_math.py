import numpy as np
from scipy.spatial.transform import Rotation

pitch_deg = 35.0
roll_deg = -20.0

r = Rotation.from_euler('xyz', [-pitch_deg, roll_deg, 0], degrees=True)
R_matrix = r.as_matrix()

n = R_matrix @ np.array([0, 0, -1])

if n[2] < 0:
    n = -n

pitch = np.degrees(np.arctan2(n[1], n[2]))
roll = np.degrees(np.arctan2(n[0], n[2]))

print(f"GT: Pitch {pitch_deg}, Roll {roll_deg}")
print(f"Est: Pitch {pitch}, Roll {roll}")

