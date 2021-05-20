import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import sobel, distance_transform_edt


def furthest_from_edge(image):
    distances = distance_transform_edt(image)
    return np.unravel_index(distances.argmax(), distances.shape)


def roi_values(array):
    return {
        "mean": array.mean(),
        "err": array.std(),
        "min": array.min(),
        "max": array.max(),
        "total": array.sum(),
    }


def make_roi_edgemaps(roi_fits, calculate_centers=True):
    roi_edgemaps = {}
    for roi_ix in roi_fits:
        roi_array = roi_fits[roi_ix].data
        name = roi_fits[roi_ix].header["EXTNAME"].lower()
        edgemap = make_edgemap(roi_array).astype("uint8")
        roi_edgemaps[name] = {"edgemap": edgemap}
        if calculate_centers:
            center_y, center_x = furthest_from_edge(roi_array)
            roi_edgemaps[name]["center"] = (center_x, center_y)
    return roi_edgemaps


def draw_edgemaps_on_image(
    image, edgemap_dict, inscribe_names=False, fontproperties=None, width=4
):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(image)
    for roi_name, edgemap in edgemap_dict.items():
        if "color" in edgemap.keys():
            color = edgemap["color"]
        else:
            color = "white"
        edge_y, edge_x = np.nonzero(edgemap["edgemap"])
        plt.scatter(
            edge_x, edge_y, marker=".", s=width, color=color, alpha=0.4
        )
        if inscribe_names:
            plt.annotate(
                roi_name,
                (edgemap["center"][0] - 15, edgemap["center"][1]),
                color="white",
                fontproperties=fontproperties,
                alpha=0.8,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def make_edgemap(image, threshold=0.01):
    edge = np.hypot(sobel(image, 0), sobel(image, 1))
    edge[np.where(edge > threshold)] = 1
    return edge / edge.max()


def count_rois_on_image(
    roi_arrays, roi_names, image, detector_mask=None, special_constants=None
):
    """
    expects ROIs 'unpacked' from their FITS format -- this is basically because
    we want applications to be able to prefilter the list without sending a
    bunch of additional information to this function, and also to count arrays
    made in other ways without rolling them into FITS files!
    """
    count_dict = {}
    if special_constants is not None:
        special_mask = np.full(image.shape, True)
        special_mask[np.isin(image, special_constants)] = False
    for roi_mask, roi_name in zip(roi_arrays, roi_names):
        assert (
            roi_mask.shape == image.shape
        ), "it seems like this ROI might have been drawn on a different image."
        if detector_mask is not None:
            roi_mask = np.logical_and(roi_mask, detector_mask)
        if special_constants is not None:
            roi_mask = np.logical_and(roi_mask, detector_mask)
        count_dict[roi_name] = roi_values(image[roi_mask])
    return count_dict


def make_roi_hdu(input_array, roi_name, metadata_dict):
    """
    make an individual HDU for a marslab .roi file
    """
    roi_hdu = fits.PrimaryHDU(input_array)
    # this parameter is separate and mandatory because we really
    # need every ROI to have some kind of distinguishing name
    roi_hdu.header["NAME"] = roi_name
    # not being strict right now
    for field, value in metadata_dict.items():
        roi_hdu.header[field] = value
    roi_hdu.name = roi_name
    return roi_hdu


def select_roi_by_ix(rois: np.ndarray, color_ix: int) -> np.ndarray:
    roi = np.zeros(shape=rois.shape, dtype=rois.dtype)
    roi[np.where(rois == color_ix)] = 1
    return roi