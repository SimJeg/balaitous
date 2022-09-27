from pathlib import Path

import torch
import numpy as np
import SimpleITK as sitk
from skimage import measure
from lungmask import mask as lungmask

from balaitous.vit import load_ibot_model


class ParseAgeSex:
    """
    Extracts PatientAge and PatientSex from metadata if 'age' and 'sex' are not already available, 
    and divide age by 100
    """
    def __call__(self, sample):
        image = sample['image']
        keys = image.GetMetaDataKeys()

        # Get age
        if sample.get('age') is None:
            if 'PatientAge' in keys:
                age = image.GetMetaData('PatientAge')
                age = float(str(age).rstrip('Y'))
            else:
                age = 65
        else:
            age = float(sample['age'])

        # Clip age and divide by 100
        sample['age'] = np.clip(age / 100., 0.25, 0.95)

        # Get sex
        if sample.get('sex') is None:
            if 'PatientSex' in keys:
                sex = image.GetMetaData('PatientSex')
                sample['sex'] = {'M': 1., 'F': 0.}.get(sex, 0.5)
            else:
                sample['sex'] = 0.5
        else:
            assert sample['sex'] in [0, 0.5, 1]
            sample['sex'] = float(sample['sex'])

        return sample


class SitkResample:
    """
    Resamples sample['image'] to a new pixel spacing with a trilinear interpolation

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
                              output_size.astype(int).tolist(), sitk.Transform(), sitk.sitkLinear, image.GetOrigin(),
                              tuple(self.output_spacing), image.GetDirection(), 0., image.GetPixelID())

        sample['image'] = image
        return sample


class CreateLungMask:
    """
    Create a segmentation mask of the lungs using the U-Net R231 model from https://github.com/JoHof/lungmask
    Mask values are 0 for background, 1 for right lung and 2 for left lung
    A new key added to the sample: 'mask'

    Parameters
    ----------
    device: str
        'cuda' to run on GPU, 'cpu' to run on CPU
    """
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        mask = lungmask.apply(sample['image'], force_cpu=self.device == 'cpu')
        mask = mask.transpose((1, 2, 0))[::-1, :, :]
        sample['mask'] = mask
        return sample


class SitkToNumpy:
    """
    Converts sample['image'] from sitk image to numpy array
    """
    def __call__(self, sample):
        image = sample['image']
        sample['pixel_spacing'] = image.GetSpacing()
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)[::-1, :, :]
        sample['image'] = image
        return sample


class CleanLungMask:
    """
    Removes spurious connected components from the lung mask:
        - keeps the two 3D biggest components for left and right lung
        - removes 2D components with 95% voxels greater than -100 HU (parameters percentile and hu_threshold)
    Converts sample['mask'] to a binary boolean mask

    Parameters
    ----------
    hu_threshold : int, optional
    percentile : int, optional
    """
    def __init__(self, hu_threshold=-100, percentile=5):
        self.hu_threshold = hu_threshold
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
                if np.percentile(image[rr, cc, i], self.percentile) > self.hu_threshold:
                    clean_mask[rr, cc, i] = False

        sample['mask'] = clean_mask
        return sample

class CropPad:
    """
    Crops or pads image and mask to output_size along axial dimension

    Parameters
    ----------
    output_size : int, optional
        output size, by default 224
    constant_values : int, optional
        constant value to use for padding, by default -1024
    """
    def __init__(self, output_size=224, constant_values=-1024):
        self.output_size = output_size
        self.value = constant_values

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        # Crop where needed (crop centered on lungs)
        for axis in [0, 1]:
            input_size = image.shape[axis]
            if input_size > self.output_size:
                other_axis = tuple([i for i in range(3) if i != axis])
                lung_center = np.nonzero(mask.sum(other_axis) > 0)[0][[0, -1]].sum() // 2
                width = self.output_size // 2
                center = np.clip(lung_center, width, input_size - width)

                image = np.take(image, range(center - width, center + width), axis=axis)
                mask = np.take(mask, range(center - width, center + width), axis=axis)

        # Pad where needed
        delta = self.output_size - np.array(image.shape)
        delta[2] = 0  # don't pad the axial dimension

        pad_left = delta // 2
        pad_right = delta - pad_left
        pad = tuple(zip(pad_left, pad_right))

        image = np.pad(image, pad, constant_values=self.value)
        mask = np.pad(mask, pad, constant_values=False)

        sample['image'] = image
        sample['mask'] = mask
        return sample


class CropLungs:
    """ 
    Crops the image to the slices containing lung and removes extreme slices along the axial dimension

    Parameters
    ----------
    percentile : int, optional
        percentile of top and bottom lung slices to remove, by default 0
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
    Sets the voxels outside the lung mask to a constant value 
    A new key added to the sample: 'unmasked_image'

    Parameters
    ----------
    constant_values : int, optional
        constant value for voxels outside of the lungs, by default -1024
    """
    def __init__(self, constant_values=-1024):
        self.value = constant_values

    def __call__(self, sample):
        sample['unmasked_image'] = sample['image'].copy()
        sample['image'][sample['mask'] == 0] = self.value
        return sample


class iBOTExtractor:
    """
    Extracts the iBOT features from the image. The features are averaged accross slices 
    and vertically flipped slices. We impose this vertical flip invariance as axial images
    can be flipped depending on DICOM processing. This does not hurt performance (it even 
    slightly improves it).

    Parameters
    ----------
    key : str
        name of the features
    checkpoint : str
        name of model to use for feature extraction, ViT-L-Imagenet of ViT-L-CT
    n_blocks : int, optional
        number of transformer layers to use, by default None
    vmin : int, optional
        minimum intensity in HU units, by default -1024
    vmax : int, optional
        maximum intensity in HU units, by default 600
    device : str, optional
        'cuda' to run on GPU, 'cpu' to run on cpu, by default None
    """
    def __init__(self, key, checkpoint, n_blocks=None, vmin=-1024, vmax=600, device=None):

        self.key = key
        self.vmin = vmin
        self.vmax = vmax
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Load model
        self.model = load_ibot_model(checkpoint, n_blocks)
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
    Applies the logistic regressions trained on the STOIC database
    """
    def __init__(self):
        weights_path = Path(__file__).resolve().parents[1] / 'assets/balaitous.npz'
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
                sample['prediction'][
                    output] += self.weights[f'alpha_{key}_{output}'] * sample['prediction'][f'{key}_{output}']

        return sample


def imagenet_preprocessing(image, vmin=-1024, vmax=600):
    """ 
    Applies ImageNet processing to the image:
        - reshapes an (H, W, Z) image into (Z, 3, W, H) 
        - rescales from [vmin, vmax] to [0, 1]
        - normalizes using ImageNet mean and standard deviation

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
