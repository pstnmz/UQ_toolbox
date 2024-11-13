import pickle
import numpy as np
import nibabel as nib
import glob
from lito_radiomics.lito_radiomics.lito_features import shape2D
import os
import cupy as cp
from cupyx.scipy import ndimage
import math
import radiomics
from radiomics import featureextractor
from math import isclose
import SimpleITK as sitk
from PIL import Image
from scipy import spatial
import scipy
from skimage.measure import EllipseModel
from skimage.draw import ellipse
import random


def check_registration(im1, im2):
    ax, az=im1.header['qoffset_x'], im1.header['qoffset_z']
    bx, bz=im2.header['qoffset_x'], im2.header['qoffset_z']

    return bool(
        math.isclose(az, bz, rel_tol=0.05)
        and math.isclose(ax, bx, rel_tol=0.05)
    )


def rescale_to_255(array, min_val, max_val):
    """
    Rescale the values of a 2D NumPy array to the range [0, 255].
    
    Parameters:
        array (numpy.ndarray): Input 2D array.
        
    Returns:
        numpy.ndarray: Rescaled array with values in the range [0, 255].
    """

    # Ensure the array is not constant to avoid division by zero
    if min_val != max_val:
        scaled_array = 255 * (array - min_val) / (max_val - min_val)
    else:
        scaled_array = array

    return scaled_array.astype(np.uint8)


def fill_holes(mask):
    return scipy.ndimage.binary_fill_holes(mask[0, :, :]).astype(float)


def resegment_thresholding(image, mask, thresh=0.2):
    
    max = (image*mask).max()
    new_mask20= np.ma.masked_where((image*mask) > thresh*max, mask)

    return new_mask20.mask.astype(float)
    

def ellipse_perimeter(a, b):
    """
    Compute the perimeter of an ellipse using Ramanujan's approximation formula.
    
    Parameters:
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
    
    Returns:
        float: Perimeter of the ellipse.
    """
    h = ((a - b) / (a + b))**2
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def compute_dice_ellipse(a, b, eigen_vect, mask_slice):
    
    angle1 = np.arctan(eigen_vect[1][1]/eigen_vect[1][0])
    angle2 = np.arctan(eigen_vect[1][1]/eigen_vect[1][0]) + np.deg2rad(90)

    center = ndimage.center_of_mass(cp.asarray(sitk.GetArrayFromImage(mask_slice)))
    xy1 = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25), params=(center[1].item(), center[0].item(), a, b, angle1))
    xy2 = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25), params=(center[1].item(), center[0].item(), a, b, angle2))
    ell1 = ellipse(center[1].item(), center[0].item(), a, b, rotation=angle1)
    ell2 = ellipse(center[1].item(), center[0].item(), a, b, rotation=angle2)

    shape_slice = sitk.GetArrayFromImage(mask_slice).shape
    mask_ellipse1 = np.zeros(shape_slice)
    mask_ellipse1[[i for (i, j) in zip(ell1[::-1][0], ell1[::-1][1]) if (i<shape_slice[0] and j<shape_slice[1])], [j for (i, j) in zip(ell1[::-1][0], ell1[::-1][1]) if (i<shape_slice[0] and j<shape_slice[1])]] = 1

    mask_ellipse2 = np.zeros(shape_slice)
    mask_ellipse2[[i for (i, j) in zip(ell2[::-1][0], ell2[::-1][1]) if (i<shape_slice[0] and j<shape_slice[1])], [j for (i, j) in zip(ell2[::-1][0], ell2[::-1][1]) if (i<shape_slice[0] and j<shape_slice[1])]] = 1

    dice1=1-spatial.distance.dice(sitk.GetArrayFromImage(mask_slice).ravel(), mask_ellipse1.ravel())
    dice2=1-spatial.distance.dice(sitk.GetArrayFromImage(mask_slice).ravel(), mask_ellipse2.ravel())
    
    return (dice1, xy1) if dice1 >= dice2 else (dice2, xy2)


def standardize_roi(image, mask, desired_area):

    # Calculate the current area of the ROI
    current_area = cp.sum(mask)
    print(f'area {str(current_area)}')
    
    if current_area > 5:
        # Calculate the scaling factor
        scale_factor = cp.sqrt(desired_area / current_area)
        print(f'mask scaling factor {str(scale_factor)}')

        return ndimage.zoom(cp.array(image), scale_factor), ndimage.zoom(cp.array(mask), scale_factor, order=0)
    else:
        return None, None
    
    
def _extracted_from_features_extraction2D_4(arg0, cut, data_spacing):
    result = sitk.GetImageFromArray(arg0[int(cut), :, :])
    result.SetSpacing(
        (
            float(data_spacing[0]),
            float(data_spacing[1]),
            float(data_spacing[2]),
        )
    )
    result = sitk.JoinSeries(result)

    return result


def _extracted_from_features_extraction2D_11():
    result = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    result.disableAllFeatures()
    result.enableFeaturesByName(
        shape2D=[
            "MajorAxisLength",
            "MinorAxisLength",
        ]
    )
    return result


def for_ellipse_creation(features_extractor, image, mask):
    results = features_extractor.execute(image, mask)
    features_lito = shape2D.LitoRadiomicsShape2D(image, mask)
    features_eigenvect = features_lito.getEigenVectorsFeatureValue()
    
    a = float(results['original_shape2D_MajorAxisLength'])/2  # Semi-major axis
    b = float(results['original_shape2D_MinorAxisLength'])/2  # Semi-minor axis
    
    return features_eigenvect, a, b


def features_extraction2D(image, mask, cut, filled=True):
    
    data_spacing = [1,1,1]
    sitk_img = _extracted_from_features_extraction2D_4(image, cut, data_spacing)
    sitk_mask = _extracted_from_features_extraction2D_4(mask, cut, data_spacing)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
    
    if filled is True:
        sitk_mask = sitk.GetImageFromArray(fill_holes(sitk.GetArrayFromImage(sitk_mask)))
    
    try:
        features_40_slice = _extracted_from_features_extraction2D_11()

        image = sitk_img[:, :, 0] if filled is True else sitk_img
        features_eigenvect, a, b = for_ellipse_creation(features_40_slice, image, sitk_mask)
        di, cont_ellipse = compute_dice_ellipse(a, b, features_eigenvect, sitk_mask)
                
        return di, cont_ellipse
    except (ValueError, TypeError):
        return None, None


def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    
    if x != 'R':
        ct_arr = np.rot90(ct_arr, axes=(0,2), k=1)
    if y != 'P':
        ct_arr = np.flip(ct_arr, axis=1)
    if z != 'S':
        ct_arr = np.flip(ct_arr, axis=2)
    
    return ct_arr


def remove_out_of_box_mask(mask, box):
    _, y, z = np.where(~box.mask)
    y_min = y.min()
    y_max = y.max()
    z_min = z.min()
    z_max = z.max()
    
    y_min_iso = int(y_max - (y_max-y_min)/2 - 25)
    y_max_iso = int(y_max - (y_max-y_min)/2 + 25)
    z_min_iso = int(z_max - (z_max-z_min)/2 - 25)
    z_max_iso = int(z_max - (z_max-z_min)/2 + 25)


    mask[:, :y_min_iso, :] = mask[:, y_max_iso+1:, :] = mask[:, :, :z_min_iso] = mask[:, :, z_max_iso+1:] = 0
    
    return mask


def add_random_rotation(im, gamma, alpha, beta):

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    
    rotation_affine_gamma = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                      [sin_gamma, cos_gamma, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
    
    rotation_affine_alpha = np.array([[cos_alpha, 0, sin_alpha, 0],
                                      [0, 1, 0, 0],
                                      [-sin_alpha, 0, cos_alpha, 0],
                                      [0, 0, 0, 1]])
    rotation_affine_beta = np.array([[1, 0, 0, 0],
                                     [0, cos_beta, -sin_beta, 0],
                                     [0, sin_beta, cos_beta, 0],
                                     [0, 0, 0, 1]])
    
    rotated_data_ax = ndimage.rotate(cp.array(im.dataobj), np.rad2deg(gamma), axes=(1, 2), reshape=True)#False)
    rotated_data_ax_cor = ndimage.rotate(rotated_data_ax, np.rad2deg(alpha), axes=(0, 2), reshape=True)#False)
    rotated_data_ax_cor_sag = ndimage.rotate(rotated_data_ax_cor, np.rad2deg(beta), axes=(0, 1), reshape=True)#False)
    im_affine_ax = im.affine.dot(rotation_affine_gamma)
    im_affine_ax_sag = im_affine_ax.dot(rotation_affine_alpha)
    im_affine_ax_sag_cor = im_affine_ax_sag.dot(rotation_affine_beta)

    return rotated_data_ax_cor_sag, im_affine_ax_sag_cor


def random_slice_selection(rotated_image, rotated_mask, rotated_bb, biggest_cut_size):
    
    min_mask = np.argwhere(rotated_mask==1)[:, 0].min()
    max_mask = np.argwhere(rotated_mask==1)[:, 0].max()
    x = random.randint(min_mask, max_mask+1)
    slice_ma_for_db = rotated_mask[x, :, :]

    if np.count_nonzero(slice_ma_for_db) > 0.05*biggest_cut_size:
        dice, cont_ellipse = features_extraction2D(rotated_image, rotated_mask, x)
    else:
        dice, cont_ellipse = None, None

    return x, dice, cont_ellipse


def create_n_instances(a_im, a_ma, a_ma_bb):
    gamma = random.uniform(np.deg2rad(0), np.deg2rad(360))
    alpha = random.uniform(np.deg2rad(0), np.deg2rad(360))
    beta = random.uniform(np.deg2rad(0), np.deg2rad(360))

    print(f'gamma: {str(np.rad2deg(gamma))}')
    print(f'alpha: {str(np.rad2deg(alpha))}')
    print(f'beta: {str(np.rad2deg(beta))}')

    a_im_rotated, rotated_affine = add_random_rotation(a_im, gamma, alpha, beta)
    a_ma_rotated, ma_rotated_affine = add_random_rotation(a_ma, gamma, alpha, beta)
    a_ma_bb_rotated, ma_bb_rotated_affine = add_random_rotation(a_ma_bb, gamma, alpha, beta)
    a_ma_rotated[a_ma_rotated > 0.1] = 1
    a_ma_rotated[a_ma_rotated <=0.1] = 0
    a_ma_bb_rotated[a_ma_bb_rotated > 0.1] = 1
    a_ma_bb_rotated[a_ma_bb_rotated <=0.1] = 0
    new_image = nib.Nifti1Image(a_im_rotated, affine=rotated_affine)
    new_mask = nib.Nifti1Image(a_ma_rotated, affine=ma_rotated_affine)
    new_mask_bb = nib.Nifti1Image(a_ma_bb_rotated, affine=ma_bb_rotated_affine)

    final_rot_image = cp.float32(new_image.dataobj.get())
    final_rot_mask = cp.float32(new_mask.dataobj.get())
    final_rot_mask_bb = cp.float32(new_mask_bb.dataobj.get())
    final_rot_mask = remove_out_of_box_mask(final_rot_mask, np.ma.masked_equal(final_rot_mask_bb, 0))

    biggest_cut = np.max([np.count_nonzero(final_rot_mask[k, :, :]) for k in range(final_rot_mask.shape[0])])

    slice_nb, dice, cont_ellipse = random_slice_selection(final_rot_image, final_rot_mask, final_rot_mask_bb, biggest_cut)

    return final_rot_image, final_rot_mask, final_rot_mask_bb, slice_nb, dice, cont_ellipse, gamma, alpha, beta


if __name__ == "__main__":
    with cp.cuda.Device(1):
        path_data_glob = "/mnt/data/psteinmetz/neotex/to_process/Ozgun/"
        list_cases = os.listdir(path_data_glob)[30:]
        log = {}
        count = 1
        for case in list_cases:
            if '.' not in case:
                print(case)
                path_data = f"{path_data_glob}{str(case)}"
                
                image_path_sub = glob.glob(
                    f'{path_data}/RawVolume/*subtracted*_Bspline_zscore.nii.gz'
                )

                mask_bb_path = glob.glob(f'{path_data}/RoiVolume/VOI_bb.nii.gz')
                mask_path = glob.glob(f'{path_data}/RoiVolume/VOI_morphoclosing.nii.gz')


                mask_bb = nib.load(mask_bb_path[0])
                mask = nib.load(mask_path[0])
       
                image = nib.load(image_path_sub[0])

                a_im = nib.Nifti1Image(check_orientation(image, np.array(image.dataobj)), affine=image.affine)
                a_ma = check_orientation(mask, np.array(mask.dataobj))
                a_ma[a_ma > 0.1] = 1
                a_ma[a_ma <=0.1] = 0
                a_ma = nib.Nifti1Image(a_ma, affine=mask.affine)
                a_ma_bb = check_orientation(mask_bb, np.array(mask_bb.dataobj))
                a_ma_bb = nib.Nifti1Image(np.ma.masked_where(a_ma_bb>0.1, a_ma_bb), affine=mask_bb.affine)
                j = 0
                for i in range(50):
                    print(f'slice {str(i)} {str(count)}/{str((len(list_cases) - 2) * 50)}')
                    final_rot_image, final_rot_mask, final_rot_mask_bb, slice_nb, dice, cont_ellipse, gamma, alpha, beta = create_n_instances(a_im, a_ma, a_ma_bb)
                    if cont_ellipse is None:
                        print('wrong slice')
                    else:
                        min_val = cp.array(a_im.dataobj).min().item()
                        max_val = cp.array(a_im.dataobj).max().item()
                        inside_bb = final_rot_mask_bb * final_rot_image
                        to_display = np.ma.masked_equal(inside_bb, 0)

                        x, y, z = np.where(~to_display.mask)

                        y_min = y.min()
                        y_max = y.max()
                        z_min = z.min()
                        z_max = z.max() 

                        y_min_iso = int(y_max - (y_max-y_min)/2 - 25)
                        y_max_iso = int(y_max - (y_max-y_min)/2 + 25)
                        z_min_iso = int(z_max - (z_max-z_min)/2 - 25)
                        z_max_iso = int(z_max - (z_max-z_min)/2 + 25)

                        im_to_save = Image.fromarray(rescale_to_255(final_rot_image[slice_nb, y_min_iso:y_max_iso+1, z_min_iso:z_max_iso+1], min_val, max_val))
                        im_to_save = im_to_save.convert("L")
                        log[f"{case}_{str(j)}"] = (dice, gamma, alpha, beta)
                        with open(f'/mnt/data/psteinmetz/neotex/data_CNN/images_Ozgun_for_calibration/Arrays/{case}_{str(j)}.npy', 'wb') as f:
                            np.save(f, final_rot_image[slice_nb, y_min_iso:y_max_iso+1, z_min_iso:z_max_iso+1])
                            np.save(f, final_rot_mask[slice_nb, y_min_iso:y_max_iso+1, z_min_iso:z_max_iso+1])

                        if dice > 0.9:
                            print('Round !')
                            im_to_save.save(
                                f"/mnt/data/psteinmetz/neotex/data_CNN/images_Ozgun_for_calibration/Round/{case}_{str(j)}"
                                + ".png"
                            )
                        elif 0.8 < dice <=0.9:
                            print('Ambiguous !')
                            im_to_save.save(
                                f"/mnt/data/psteinmetz/neotex/data_CNN/images_Ozgun_for_calibration/Ambiguous/{case}_{str(j)}"
                                + ".png"
                            )
                        else:
                            print('Irregular !')
                            im_to_save.save(
                                f"/mnt/data/psteinmetz/neotex/data_CNN/images_Ozgun_for_calibration/Irregular/{case}_{str(j)}"
                                + ".png"
                            )

                        j = j+1
                    count = count+1
        with open('/mnt/data/psteinmetz/neotex/data_CNN/images_Ozgun_for_calibration/log1.pickle', 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)