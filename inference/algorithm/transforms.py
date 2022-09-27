import numpy as np
import SimpleITK as sitk
from pathlib import Path
import torch
from skimage import measure
from lungmask import mask as lungmask
from lungmask.mask import get_model

from algorithm.vit import load_ibot_model


class ParseAgeSex:
    """
    Extract PatientAge and PatientSex from metadata
    age default value is 65 and sex default value is 0.5 (1 for male, 0 for female)
    age is then divided by 100
    """

    def __call__(self, sample):
        image = sample['image']
        keys = image.GetMetaDataKeys()

        if 'PatientAge' in keys:
            age = image.GetMetaData('PatientAge')
            if str(age)[-1] == 'Y':
                age = age[:-1]
            try:
                age = float(age)
            except:
                age = 65
        else:
            age = 65  # median of the STOIC dataset
        sample['age'] = np.clip(age / 100, 0.25, 0.95)

        if 'PatientSex' in keys:
            sex = image.GetMetaData('PatientSex')
            sex = {'M': 1., 'F': 0.}.get(sex, 0.5)
        else:
            sex = 0.5
        sample['sex'] = sex

        return sample


class SitkResample:
    """
    Resample sample['image'] to a new pixel spacing with a trilinear interpolation

    Parameters
    ----------
        output_spacing: tuple(float)
            new spacing in mm/voxel
    """

    def __init__(self, output_spacing):
        self.output_spacing = np.array(output_spacing).astype(float)

    def __call__(self, sample):
        image = sample['image']
        input_spacing = np.array(image.GetSpacing())
        input_size = np.array(image.GetSize())
        output_size = input_size * input_spacing / self.output_spacing

        image = sitk.Resample(image,
                              output_size.astype(int).tolist(),
                              sitk.Transform(), sitk.sitkLinear,
                              image.GetOrigin(), tuple(self.output_spacing),
                              image.GetDirection(), 0., image.GetPixelID())

        sample['image'] = image
        return sample


class CreateLungMask:
    """
    Create a segmentation mask of the lungs using the U-Net R231 model
    Mask values are 0 for background, 1 for right lung and 2 for left lung
    New key: mask

    Parameters
    ----------
        device: str
            'cuda' to run on GPU, 'cpu' to run on CPU
    """

    def __init__(self, device):
        self.device = device
        modelpath = Path(__file__).resolve().parents[1] / 'artifact/unet_r231-d5d2fc3d.pth'
        self.model = get_model('unet', 'R231', modelpath=modelpath)

    def __call__(self, sample):
        mask = lungmask.apply(sample['image'], model=self.model, force_cpu=self.device == 'cpu')
        mask = mask.transpose((1, 2, 0))[::-1, :, :]
        sample['mask'] = mask
        return sample


class SitkToNumpy:
    """
    Convert sample['image'] from sitk image to numpy array
    """

    def __call__(self, sample):
        image = sample['image']
        sample['pixel_spacing'] = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)[::-1, :, :]
        sample['image'] = image
        return sample


class CleanLungMask:
    """
    Remove spurious connected components from the lung mask:
        - only keeps the two 3D biggest components for left and right lung
        - deletes 2D components with 95% voxels greater than -100 HU are removed
    Finally converts sample['mask'] to a binary boolean mask

    Parameters
    ----------
        hu_threshold: int
        percentile: int
    """

    def __init__(self, hu_threshold=-100, percentile=5):
        self.threshold = hu_threshold
        self.percentile = percentile

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        clean_mask = np.zeros(mask.shape, dtype=bool)

        # For left and right lung, only keep the biggest connected component
        for i in [1, 2]:
            label_mask = measure.label(mask == i)
            regions = measure.regionprops(label_mask)
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
            if len(regions):
                clean_mask += (label_mask == regions[0].label)

        # For each slice, only keep components with low enough HU units
        for i in range(mask.shape[2]):
            label_mask_i = measure.label(clean_mask[..., i])
            regions = measure.regionprops(label_mask_i)
            for r in regions:
                rr, cc = r.coords.T
                if np.percentile(image[rr, cc, i],
                                 self.percentile) > self.threshold:
                    clean_mask[rr, cc, i] = False

        sample['mask'] = clean_mask
        return sample


class CropLungs:
    """ 
    Crop image to slices containing lung and remove extreme slices along the axial dimension

    Parameters
    ----------
        percentile: int. If 10, then the 10% of the top and bottom slices are removed
    """

    def __init__(self, percentile=0):
        self.percentile = percentile / 100.

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # Remove slices without lungs
        start, end = np.nonzero(mask.sum((0, 1)) > 0)[0][[0, -1]]
        image = image[..., start:end]
        mask = mask[..., start:end]

        # Remove extreme slices
        if self.percentile > 0:
            n = image.shape[2]
            start = int(n * self.percentile)
            end = int(n * (1 - self.percentile))
            image = image[..., start:end]
            mask = mask[..., start:end]

        sample['image'] = image
        sample['mask'] = mask

        return sample


class ApplyLungMask:
    """ 
    Set the voxels outside the lung mask to a constant value for each key beginning with 'image'
    """

    def __init__(self, constant_values=-1024):
        self.value = constant_values

    def __call__(self, sample):
        sample['image'][sample['mask'] == 0] = self.value
        return sample


class CropPad:
    """
    Crop or pad image and mask to output_size along axial axis

    Parameters
    ----------
        output_size: int
        constant_values: int
    """

    def __init__(self, output_size=224, constant_values=-1024):
        self.output_size = output_size
        self.value = constant_values

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # Crop where needed
        for axis in [0, 1]:
            input_size = image.shape[axis]
            if input_size > self.output_size:
                # If the image it too big, crop, centering on the lung mask
                other_axis = tuple([i for i in range(3) if i != axis])
                lung_center = np.nonzero(
                    mask.sum(other_axis) > 0)[0][[0, -1]].sum() // 2
                width = self.output_size // 2
                center = np.clip(lung_center, width, input_size - width)

                image = np.take(image, range(center - width, center + width), axis=axis)
                mask = np.take(mask, range(center - width, center + width), axis=axis)

        # Pad where needed
        delta = self.output_size - np.array(image.shape)
        delta[2] = 0  # don't pad n the axial dimension

        pad_left = delta // 2
        pad_right = delta - pad_left
        pad = tuple(zip(pad_left, pad_right))

        image = np.pad(image, pad, constant_values=self.value)
        mask = np.pad(mask, pad, constant_values=False)

        sample['image'] = image
        sample['mask'] = mask
        return sample


class iBOTExtractor:
    """
    Extract the iBOT features from the image. The features are averaged accross slices 
    and vertically flipped slices. We impose a vertical flip invariance as axial images
    can be flipped depending on DICOM processing. 

    Parameters
    ----------
        key: str. name of the features
        checkpoint: str. name of model to use for feature extraction
        n_blocks: int. number of transformer layers to use
        vmin: int. minimum value of the image in HU units
        vmax: int. maximum value of the image in HU units
        device: str
    """

    def __init__(self, key, checkpoint, n_blocks=None, vmin=-1024, vmax=600, device=None):

        self.key = key
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = load_ibot_model(checkpoint, n_blocks)
        self.vmin = vmin
        self.vmax = vmax

        self.device = device
        self.model.eval()
        self.model.to(device)

    def __call__(self, sample):

        # Prepare input tensors
        image = sample['image'].copy()
        image = imagenet_preprocessing(image, vmin=self.vmin, vmax=self.vmax)
        image = torch.from_numpy(image).type(torch.FloatTensor).to(self.device)
        flipped_image = torch.flip(image, dims=(2, ))

        # Extract features
        feature_list = []
        for tensor in [image, flipped_image]:
            with torch.no_grad():
                tokens = self.model.prepare_tokens(tensor)
                for b in self.model.blocks:
                    tokens = b(tokens)
            features = tokens[:, 0]  # class token
            features = features.mean(0)  # average pooling across slices
            features = features.detach().cpu().numpy()
            feature_list.append(features)

        sample.setdefault('features', {})[self.key] = np.mean(feature_list, axis=0)

        return sample


class ApplySTOICLogisticRegressions:
    """
    Apply the logistic regressions trained on the STOIC database
    """

    def __init__(self):
        weights_path = Path(__file__).resolve().parents[1] / 'artifact/balaitous.npz'
        self.weights = np.load(weights_path)

    def __call__(self, sample):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for output in ['covid', 'severe']:
            for key in ['full', 'lung']:
                features = sample['features'][key]
                if output == 'severe':
                    # Add age and sex for severity prediction
                    features = np.hstack([features, [sample['age'], sample['sex']]])

                # Apply logistic regression
                coef = self.weights[f'coef_{key}_{output}']
                intercept = self.weights[f'intercept_{key}_{output}']
                prediction = sigmoid(np.dot(features, coef) + intercept)
                sample.setdefault('prediction', {})[f'{key}_{output}'] = prediction

        # Average predictions
        for output in ['severe', 'covid']:
            sample['prediction'][output] = 0
            for key in ['full', 'lung']:
                sample['prediction'][output] += self.weights[f'alpha_{key}_{output}'] * sample['prediction'][f'{key}_{output}']

        return sample


def imagenet_preprocessing(image, vmin=-1024, vmax=600):
    """ Apply ImageNet processing to the image:
        - reshape an (H, W, Z) image into (Z, 3, W, H) 
        - rescale from [vmin, vmax] to [0, 1]
        - normalize using ImageNet mean and standard deviation

    Parameters
    ----------
    image : np.ndarray
        input image
    vmin : int, optional
        minimum HU value, by default -1024
    vmax : int, optional
        maximum HU value, by default 600

    Returns
    -------
    np.ndarray
        output image
    """

    # Reshape
    image = image.transpose((2, 0, 1))[:, None].repeat(3, axis=1)

    # Rescale
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin)

    # Normalize with ImageNet mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean.reshape(1, 3, 1, 1)) / std.reshape(((1, 3, 1, 1)))

    return image
