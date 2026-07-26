from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import csv as pcsv
import spiceypy as spice
from spiceypy import cyice

KERNEL_PATH = Path(__file__).parent.parent.parent / "kernels"


def set_up_kernels(include_ch1_kernel: bool = True) -> None:
    # ch1 SCLK kernel
    spice.furnsh(str(KERNEL_PATH / "aig_ch1_sclk_complete_biased_m1p816.tsc"))
    # basic leapseconds kernel
    spice.furnsh(str(KERNEL_PATH / "naif0012.tls"))
    # general planetary ephemeris kernel
    spice.furnsh(str(KERNEL_PATH / "de442s.bsp"))
    # MOON_PA / MOON_ME frame definition kernel
    spice.furnsh(str(KERNEL_PATH / "moon_de440_250416.tf"))
    # high-precision lunar frame orientation pck
    spice.furnsh(str(KERNEL_PATH / "moon_pa_de440_200625.bpc"))
    if include_ch1_kernel:
        spice.furnsh(
            str(KERNEL_PATH / "ch-1-jpl-merged-23-march-2010-1220.bsp")
        )


def get_line_times_et_from_l1b(l1b_time_file: Path) -> np.ndarray:
    """Get per-line times in ET from an L1B time file."""
    l1b_time = pcsv.read_csv(
        l1b_time_file,
        read_options=pcsv.ReadOptions(autogenerate_column_names=True),
    )
    return cyice.utc2et(l1b_time["f1"].to_numpy().astype(str))


def as_et_array(times: np.ndarray | float) -> np.ndarray:
    """Return ET values as a contiguous one-dimensional float64 array."""
    ets = np.asarray(times, dtype=np.double)
    if ets.ndim == 0:
        ets = ets.reshape(1)
    elif ets.ndim != 1:
        raise ValueError(
            f"Expected scalar or 1-D ET array, got shape {ets.shape}."
        )
    return np.ascontiguousarray(ets)


def unit(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize vectors along `axis`."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    if np.any(norm < 1e-15):
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def Rx(angle_rad: float | np.ndarray) -> np.ndarray:
    """Active right-handed rotation about +x, optionally batched."""
    angle_rad = np.asarray(angle_rad, dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    matrix = np.zeros(angle_rad.shape + (3, 3), dtype=float)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = c
    matrix[..., 1, 2] = -s
    matrix[..., 2, 1] = s
    matrix[..., 2, 2] = c
    return matrix


def Ry(angle_rad: float | np.ndarray) -> np.ndarray:
    """Active right-handed rotation about +y, optionally batched."""
    angle_rad = np.asarray(angle_rad, dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    matrix = np.zeros(angle_rad.shape + (3, 3), dtype=float)
    matrix[..., 0, 0] = c
    matrix[..., 0, 2] = s
    matrix[..., 1, 1] = 1.0
    matrix[..., 2, 0] = -s
    matrix[..., 2, 2] = c
    return matrix


def Rz(angle_rad: float | np.ndarray) -> np.ndarray:
    """Active right-handed rotation about +z, optionally batched."""
    angle_rad = np.asarray(angle_rad, dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    matrix = np.zeros(angle_rad.shape + (3, 3), dtype=float)
    matrix[..., 0, 0] = c
    matrix[..., 0, 1] = -s
    matrix[..., 1, 0] = s
    matrix[..., 1, 1] = c
    matrix[..., 2, 2] = 1.0
    return matrix


def axis_angle_rotation(
    axis: np.ndarray,
    angle_rad: float | np.ndarray,
) -> np.ndarray:
    """
    Active right-handed rotation about a fixed axis, optionally for many angles.

    Returns shape (3, 3) for a scalar angle or (..., 3, 3) for an array of
    angles. The axis and rotated vectors are expressed in the same frame.
    """
    axis = unit(np.asarray(axis, dtype=float))
    angle_rad = np.asarray(angle_rad, dtype=float)

    ux, uy, uz = axis
    # K @ v == axis X v
    K = np.array(
        [
            [0.0, -uz, uy],
            [uz, 0.0, -ux],
            [-uy, ux, 0.0],
        ]
    )

    sin_angle = np.sin(angle_rad)[..., None, None]
    one_minus_cos = (1.0 - np.cos(angle_rad))[..., None, None]

    return np.eye(3) + sin_angle * K + one_minus_cos * (K @ K)


def ch1_position_velocity_j2000(
    times: np.ndarray | float,
    abcorr: str = "NONE",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Chandrayaan-1 state relative to the Moon in J2000 for all times.

    Position is Moon -> CH1. Both returned arrays have shape (n_times, 3).
    """
    ets = as_et_array(times)

    # Explicitly select Cyice's vectorized implementation. `states` has
    # shape (n_times, 6), and light times have shape (n_times,).
    states, _light_times = cyice.spkez_v(
        -86,       # CH1
        ets,
        "J2000",
        abcorr,    # "NONE", "LT", or "LT+S"
        301,       # Moon
    )

    states = np.asarray(states, dtype=float)
    return states[:, :3], states[:, 3:]


def ch1_orbit_frame_j2000(
    times: np.ndarray | float,
    abcorr: str = "NONE",
) -> np.ndarray:
    """
    Return c_j2000_from_orbit for each ET.

    The result has shape (n_times, 3, 3). For every matrix, the columns are
    the instantaneous ideal orbit-frame axes expressed in J2000:

        column 0: +x, along transverse velocity
        column 1: +y, completing a right-handed frame
        column 2: +z, spacecraft -> selenocenter

    Consequently, for each time i:

        vector_j2000 = matrices[i] @ vector_orbit
        vector_orbit = matrices[i].T @ vector_j2000
    """
    position, velocity = ch1_position_velocity_j2000(times, abcorr=abcorr)

    # SPICE position is Moon -> CH1, so nadir is the opposite direction.
    z_axis = -unit(position)

    # Remove the component of velocity parallel to nadir.
    transverse_velocity = velocity - (
        np.sum(velocity * z_axis, axis=-1, keepdims=True) * z_axis
    )

    transverse_speed = np.linalg.norm(transverse_velocity, axis=-1)
    bad = np.flatnonzero(transverse_speed < 1e-12)
    if bad.size:
        raise ValueError(
            "Cannot define orbit-frame +x: velocity is nearly radial "
            f"at time indices {bad.tolist()}."
        )

    x_axis = unit(transverse_velocity)

    # x, y, z is right-handed because x cross y = z.
    y_axis = unit(np.cross(z_axis, x_axis))

    # Remove residual floating-point non-orthogonality.
    x_axis = unit(np.cross(y_axis, z_axis))

    # Stack as columns, not rows.
    return np.stack((x_axis, y_axis, z_axis), axis=-1)


def orbit_from_ch1_model1(
    roll_deg: float | np.ndarray,
    pitch_deg: float | np.ndarray,
    yaw_deg: float | np.ndarray,
) -> np.ndarray:
    """
    Return c_orbit_from_ch1 using the plausible Model 1 convention:

        c_orbit_from_ch1 = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This convention remains an interpretation because the SIS does
    not explicitly state the Euler composition order.

    Scalar inputs return (3, 3). One-dimensional angle arrays return
    (n_times, 3, 3).
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def ch1_j2000_model1(
    times: np.ndarray | float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    abcorr: str = "NONE",
) -> np.ndarray:
    """
    Return c_j2000_from_ch1 for every ET in `times` using attitude model 1.

    The result has shape (n_times, 3, 3). Columns are CH1 +x, +y, and +z
    expressed in J2000.
    """
    j2000_from_orbit = ch1_orbit_frame_j2000(times, abcorr=abcorr)
    orbit_from_ch1 = orbit_from_ch1_model1(roll_deg, pitch_deg, yaw_deg)
    return j2000_from_orbit @ orbit_from_ch1


def ch1_j2000_model2(
    times: np.ndarray | float,
    et0: float,
    initial_roll_deg: float,
    initial_pitch_deg: float,
    initial_yaw_deg: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    abcorr: str = "NONE",
) -> np.ndarray:
    """
    Return c_j2000_from_ch1 for every ET in `times` using attitude model 2.

    Rates are degrees per second. The result has shape (n_times, 3, 3).
    """
    ets = as_et_array(times)
    dt = ets - et0

    roll_deg = initial_roll_deg + dt * roll_rate
    pitch_deg = initial_pitch_deg + dt * pitch_rate
    yaw_deg = initial_yaw_deg + dt * yaw_rate

    j2000_from_orbit = ch1_orbit_frame_j2000(ets, abcorr=abcorr)
    orbit_from_ch1 = orbit_from_ch1_model1(roll_deg, pitch_deg, yaw_deg)
    return j2000_from_orbit @ orbit_from_ch1


def setup_model_3_args(
    et0: float,
    initial_roll_deg: float,
    initial_pitch_deg: float,
    initial_yaw_deg: float,
    rotation_x_component: float,
    rotation_y_component: float,
    rotation_z_component: float,
    abcorr: str = "NONE",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute time-invariant quantities for model 3.

    Returns:
        axis_j2000: shape (3,)
        j2000_from_ch1_t0: shape (3, 3)
    """
    j2000_from_orbit_t0 = ch1_orbit_frame_j2000(
        np.array([et0], dtype=np.double),
        abcorr=abcorr,
    )[0]

    orbit_t0_from_ch1_t0 = orbit_from_ch1_model1(
        initial_roll_deg,
        initial_pitch_deg,
        initial_yaw_deg,
    )

    j2000_from_ch1_t0 = j2000_from_orbit_t0 @ orbit_t0_from_ch1_t0

    rotation_axis_orbit_t0 = np.array(
        [rotation_x_component, rotation_y_component, rotation_z_component],
        dtype=float,
    )

    axis_j2000 = unit(
        j2000_from_orbit_t0 @ unit(rotation_axis_orbit_t0)
    )
    return axis_j2000, j2000_from_ch1_t0


def ch1_j2000_model3(
    times: np.ndarray | float,
    et0: float,
    axis_j2000: np.ndarray,
    j2000_from_ch1_t0: np.ndarray,
    rotation_rate_deg_s: float,
) -> np.ndarray:
    """
    Return c_j2000_from_ch1 for every ET in `times` using attitude model 3.

    This differs from the definition in the SIS: despite the SIS calling
    the rotation-axis components J2000 XYZ, the numerical behavior suggests
    that they are components in ideal CH1 orbit frame at T0.

    This is written to use derived quantities in order to minimize per-line
    computations: see setup_model_3_args().
    """
    ets = as_et_array(times)
    angle_rad = np.deg2rad(rotation_rate_deg_s * (ets - et0))

    j2000_rotation_from_t0 = axis_angle_rotation(axis_j2000, angle_rad)
    return j2000_rotation_from_t0 @ j2000_from_ch1_t0


def ch1_j2000_for_obs(
    obsid: str,
    orientation_metadata: pd.DataFrame,
    times: np.ndarray | float,
    abcorr: str = "NONE",
) -> np.ndarray:
    """
    Return c_j2000_from_ch1 for an observation at all requested ETs.

    The output shape is (n_times, 3, 3). For each time:

        matrices[i, :, 0] is CH1 +x in J2000
        matrices[i, :, 1] is CH1 +y in J2000
        matrices[i, :, 2] is CH1 +z/down in J2000
    """
    ets = as_et_array(times)

    try:
        row = orientation_metadata.loc[
            orientation_metadata["obsid"] == obsid.upper()
        ].iloc[0]
    except IndexError as exc:
        raise ValueError(f"{obsid} not found in table.") from exc

    orientation_model = int(row["orientation_model"])

    if orientation_model == 1:
        return ch1_j2000_model1(
            ets,
            row["roll"],
            row["pitch"],
            row["yaw"],
            abcorr=abcorr,
        )

    if orientation_model == 2:
        return ch1_j2000_model2(
            ets,
            row["orientation_epoch_time"],
            row["roll"],
            row["pitch"],
            row["yaw"],
            row["roll_rate"],
            row["pitch_rate"],
            row["yaw_rate"],
            abcorr=abcorr,
        )

    if orientation_model == 3:
        et0 = row["orientation_epoch_time"]
        axis_j2000, j2000_from_ch1_t0 = setup_model_3_args(
            et0,
            row["roll"],
            row["pitch"],
            row["yaw"],
            row["x_unit"],
            row["y_unit"],
            row["z_unit"],
            abcorr=abcorr,
        )

        return ch1_j2000_model3(
            ets,
            et0,
            axis_j2000,
            j2000_from_ch1_t0,
            row["spacecraft_rotation_rate"],
        )

    raise ValueError(f"Unknown orientation model {orientation_model}")


def ch1_down_j2000_for_obs(
    obsid: str,
    orientation_metadata: pd.DataFrame,
    times: np.ndarray | float,
    abcorr: str = "NONE",
) -> np.ndarray:
    """Helper returning only CH1 +z/down in J2000."""
    return ch1_j2000_for_obs(
        obsid,
        orientation_metadata,
        times,
        abcorr=abcorr,
    )[..., 2]
