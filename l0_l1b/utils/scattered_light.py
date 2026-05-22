import numpy as np


def basic_scattered_light_correction(
        obs_image: np.ndarray,
        left_cols: list,
        right_cols: list,
        all_cols: list,
) -> np.ndarray:
    """
    Basic scattered light correction using the ratio between the vignetted
    columns on the left and right side and the median col values

    Green describes this as "additive" so I guess the idea is to add the light
    that was scattered back to the image. So I think we use the ratio between
    the vignetted columns and the median values of the actual image to multiply
    each pixel by?

    """
    num_frames, num_channels, num_cols_total = obs_image.shape

    # area we're doing correction for
    start_col = np.max(left_cols)
    end_col = np.min(right_cols)
    num_cols = end_col - start_col

    # weight each col more heavily by the side of the detector it's closer too
    col_weights = np.linspace(0, 1, num_cols)

    # we have to do this frame by frame to save memory
    for frame_ix in range(num_frames):
        frame = obs_image[frame_ix]

        median_light = np.median(frame[:, all_cols], axis=1)

        # extra buffer just in case
        median_center = np.median(frame[:, start_col+2:end_col-2])

        ratios = median_light / median_center

        obs_image[frame_ix, :, start_col:end_col] += ratios[:, np.newaxis] * \
                                                     obs_image[
                                                     frame_ix,
                                                     :,
                                                     start_col:end_col
                                                     ]
    from astropy.io import fits

    fits.writeto(
        f"lastframe_ratios_sl.fits",
        ratios[:, np.newaxis] * \
        obs_image[
        frame_ix,
        :,
        start_col:end_col
        ],
        overwrite=True
    )


    return obs_image

#
# def basic_scattered_light_correction(
#         obs_image: np.ndarray,
#         left_cols: list,
#         right_cols: list,
# ) -> np.ndarray:
#     """
#     Basic scattered light correction using the ratio between the vignetted
#     columns on the left and right side and the median col values
#
#     Green describes this as "additive" so I guess the idea is to add the light
#     that was scattered back to the image. So I think we use the ratio between
#     the vignetted columns and the median values of the actual image to multiply
#     each pixel by?
#
#     """
#     num_frames, num_channels, num_cols_total = obs_image.shape
#
#     # area we're doing correction for
#     start_col = np.max(left_cols)
#     end_col = np.min(right_cols)
#     num_cols = end_col - start_col
#
#     # weight each col more heavily by the side of the detector it's closer too
#     col_weights = np.linspace(0, 1, num_cols)
#
#     # we have to do this frame by frame to save memory
#     for frame_ix in range(num_frames):
#         frame = obs_image[frame_ix]
#
#         median_left_light = np.median(frame[:, left_cols], axis=1)
#         median_right_light = np.median(frame[:, right_cols], axis=1)
#
#         median_center = np.median(frame[:, start_col+1:end_col], axis=1)
#
#         left_ratios = median_left_light / median_center
#         right_ratios = median_right_light / median_center
#
#         interpolated_light = (
#             left_ratios[:, np.newaxis] * (1 - col_weights) +
#             right_ratios[:, np.newaxis] * col_weights
#         )
#
#         obs_image[frame_ix, :, start_col:end_col] += interpolated_light *\
#                                                       obs_image[
#                                                       frame_ix,
#                                                       :,
#                                                       start_col:end_col
#                                                       ]
#     return obs_image


# def basic_scattered_light_correction(
#         obs_image: np.ndarray,
#         left_cols: list,
#         right_cols: list,
#         all_cols: list,
# ) -> np.ndarray:
#     """
#     Basic scattered light correction using the ratio between the vignetted
#     columns on the left and right side and the median col values
#
#     I don't understand why but this is "applied on a line by line" basis (aka
#     frame by frame). I think that has got to be wrong.
#     """
#     num_frames, num_channels, num_cols_total = obs_image.shape
#
#     # area we're doing correction for
#     start_col = np.max(left_cols)
#     end_col = np.min(right_cols)
#
#     for frame_ix in range(num_frames):
#         frame = obs_image[frame_ix]
#
#         mean_light = np.median(frame[:, all_cols], axis=1)
#         obs_image[frame_ix, :, start_col:end_col] += mean_light[:, np.newaxis]
#     return obs_image

