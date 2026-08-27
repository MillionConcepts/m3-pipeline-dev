import numpy as np

# ratios derived from median of sl ratios from multiple observations where
# sl ratio is left sl column divided by median illumination in detector center

global_sl_ratios = [0.2229, 0.1605, 0.0835, 0.0532, 0.0451, 0.0502, 0.0495,
                    0.0399, 0.041, 0.0359, 0.0456, 0.0604, 0.0571, 0.0339,
                    0.0373, 0.0341, 0.0344, 0.0203, 0.0181, 0.0097, 0.0049,
                    0.003, 0.0033, 0.0, 0.0033, -0.0044, -0.0051, -0.0033,
                    -0.0024, -0.001, -0.0059, -0.0028, -0.0075, -0.0023,
                    -0.0032, -0.003, -0.0017, -0.0017, -0.0013, -0.0032,
                    -0.0013, -0.0003, -0.003, -0.001, 0.0008, 0.0012, 0.0,
                    -0.0054, -0.0013, -0.0045, -0.011, -0.0115, -0.0111,
                    -0.0116, -0.0116, -0.008, -0.0095, -0.0091, -0.0093, -0.01,
                    -0.0102, -0.0095, -0.0083, -0.0085, -0.0088, -0.0084,
                    -0.0079, -0.0094, -0.0108, -0.0082, -0.0096, -0.0101,
                    -0.0088, -0.0092, -0.0096, -0.0096, -0.0102, -0.0115,
                    -0.0077, -0.0079, -0.008, -0.0086, -0.008, -0.0082, -0.01,
                    0.0352]

target_sl_ratios = [0.4079, 0.3222, 0.3174, 0.2782, 0.2335, 0.2446, 0.1797,
                    0.2003, 0.1725, 0.1571, 0.1264, 0.1211, 0.1102, 0.0891,
                    0.0876, 0.0815, 0.1001, 0.0883, 0.0749, 0.0918, 0.0834,
                    0.0915, 0.0957, 0.0835, 0.1047, 0.096, 0.0976, 0.0921,
                    0.0872, 0.0787, 0.0937, 0.0888, 0.0782, 0.0768, 0.0775,
                    0.0772, 0.1063, 0.0821, 0.0973, 0.1197, 0.1455, 0.1002,
                    0.0927, 0.0659, 0.0848, 0.0824, 0.0767, 0.0803, 0.0709,
                    0.0771, 0.0594, 0.0585, 0.0576, 0.0512, 0.0494, 0.0383,
                    0.0405, 0.0336, 0.0336, 0.0288, 0.029, 0.0354, 0.0268,
                    0.0319, 0.032, 0.0267, 0.0199, 0.0216, 0.0206, 0.0202,
                    0.0198, 0.0202, 0.0192, 0.0209, 0.02, 0.0224, 0.0192,
                    0.021, 0.0174, 0.0228, 0.0194, 0.0203, 0.0178, 0.0205,
                    0.0209, 0.0168, 0.0166, 0.0205, 0.0196, 0.0244, 0.0171,
                    0.0304, 0.0202, 0.0206, 0.0206, 0.0222, 0.0253, 0.0274,
                    0.0265, 0.0222, 0.0264, 0.0212, 0.0307, 0.0241, 0.0232,
                    0.0279, 0.0277, 0.0345, 0.028, 0.0262, 0.0239, 0.0201,
                    0.0211, 0.025, 0.0285, 0.0194, 0.0016, 0.0131, 0.0189,
                    0.0174, 0.0132, 0.0118, 0.0113, 0.0125, 0.0107, 0.0094,
                    0.0094, 0.0121, 0.0083, -0.1081, 0.0092, 0.0101, -0.001,
                    0.0084, 0.0123, 0.0144, 0.0125, 0.0137, 0.012, 0.0177,
                    0.0119, 0.0125, 0.0107, 0.0138, 0.011, 0.0109, 0.0135,
                    0.0091, 0.0126, 0.0117, 0.0119, 0.0129, 0.0079, 0.0101,
                    0.0105, 0.0107, 0.0137, 0.0112, 0.0105, 0.0119, 0.0112,
                    0.0118, 0.0112, 0.0108, 0.009, 0.0098, 0.012, 0.0101,
                    0.0124, 0.0123, 0.0107, 0.0121, 0.0104, 0.01, 0.0137,
                    0.0148, 0.0092, 0.0091, 0.013, 0.0085, 0.0112, 0.0121,
                    0.0097, 0.0145, 0.0132, 0.0115, 0.0092, 0.0125, 0.0096,
                    0.0104, 0.0117, -0.0019, 0.013, 0.0127, 0.0129, 0.0106,
                    0.0093, 0.0145, 0.0093, 0.0123, 0.0143, 0.0117, 0.0105,
                    0.0103, 0.0095, 0.0113, 0.0099, 0.0083, 0.0108, 0.0113,
                    0.009, 0.0114, 0.0129, 0.0104, 0.0087, 0.0098, 0.0103,
                    0.0118, 0.0138, 0.0103, 0.0083, 0.007, 0.007, 0.0085,
                    0.0086, 0.0062, 0.0062, 0.0098, 0.0115, 0.0131, 0.0141,
                    0.015, 0.0128, 0.0104, 0.0141, 0.0109, 0.0099, 0.0138,
                    0.0153, 0.0127, 0.0166, 0.0099, 0.0123, 0.0097, 0.0113,
                    0.0096, 0.0172, 0.0099, 0.0147, 0.0167, 0.0124, 0.0069,
                    0.0105, 0.0079, 0.0079, 0.013, 0.0096, 0.0111, 0.0152,
                    4.2265]


def basic_kernel_scattered_light_corr(
        obs_band: np.ndarray,
        sl_ratio: float,
        sigma: int = 6,
):
    """
    Gaussian kernel applied per line per channel, scaled by the scattered light
    ratio per channel. Resulting image then scaled to retain removed scattered
    signal in peak areas.

    Args:
        obs_band: Obs image data, per channel.
        sl_ratio: Percent observed signal that is scattered light, based on
            signal in vignetted columns vs observing columns. We could evaluate
            this per line? But right now it's a set value for each band.
        sigma: Sigma for Gaussian kernel.
    """
    from scipy.ndimage import gaussian_filter1d

    # to take the image back to all signal retained after sl subtraction
    norm_factor = 1.0 / (1.0 - sl_ratio)
    scatter = gaussian_filter1d(obs_band, sigma=sigma, axis=1)
    background = sl_ratio * scatter
    obs_band = (obs_band - background) * norm_factor

    return obs_band


def basic_scattered_light_corr(
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
        median_center = np.median(frame[:, start_col + 2:end_col - 2])

        ratios = median_light / median_center

        obs_image[frame_ix, :, start_col:end_col] += ratios[:, np.newaxis] * \
                                                     obs_image[
                                                     frame_ix,
                                                     :,
                                                     start_col:end_col
                                                     ]
    # from astropy.io import fits
    # fits.writeto(
    #     f"lastframe_ratios_sl.fits",
    #     ratios[:, np.newaxis] * \
    #     obs_image[
    #     frame_ix,
    #     :,
    #     start_col:end_col
    #     ],
    #     overwrite=True
    # )

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
