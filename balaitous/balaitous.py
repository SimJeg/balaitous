import torch
from torchvision.transforms import Compose
import SimpleITK as sitk

from balaitous.transforms import *


class Balaitous:
    def __init__(self, device=None):
        """
        Given an input CT scan, Balaitous outputs a probability for COVID  disease and a probability 
        for severe outcome, defined as intubation or death within one month.

        Parameters
        ----------
        device : str, optional
            'cuda' to run on GPU, 'cpu' to run on cpu, by default None
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # We extend the classical use of transforms to all the processing steps (not only image preprocessing)
        self.transforms = Compose([
            ParseAgeSex(),
            SitkResample(output_spacing=(1.5, 1.5, 5)),
            CreateLungMask(device),
            SitkToNumpy(),
            CleanLungMask(hu_threshold=-100, percentile=5),
            CropPad(output_size=224, constant_values=-1024),
            CropLungs(percentile=10),
            iBOTExtractor(key='full', checkpoint='ViT-L-CT', n_blocks=24, vmin=-1024, vmax=600, device=device),
            ApplyLungMask(constant_values=-1024),
            iBOTExtractor(key='lung', checkpoint='ViT-L-ImageNet', n_blocks=16, vmin=-1024, vmax=600, device=device),
            ApplySTOICLogisticRegressions()
        ])

    def __call__(self, image_path, age=None, sex=None, return_dict=False):
        """
        Runs the Balaitous model and outputs a probability for COVID disease and for severe outcome.

        If not provided, age and sex are automatically parsed from the PatientAge and PatientSex metadata keys of 
        the image. If not available, default values are used (age=65, sex=0.5).

        Parameters
        ----------
        image_path : str
            path to the input image
        age : int, optional
            age of the patient, by default None
        sex : int, optional
            sex of the patient (1 for male, 0 for female), by default None
        return_dict : bool, optional
            if True, returns a dictionnary with intermediate transforms outputs, by default False

        Returns
        -------
        tuple or dict
            if return_dict is False, returns a tuple of 
            (probability of COVID, probability of severe outcome)
        """

        image = sitk.ReadImage(str(image_path))
        sample = {'image_path': image_path, 'image': image, 'age': age, 'sex': sex}
        sample = self.transforms(sample)

        if return_dict:
            return sample
        else:
            p_covid = sample['prediction']['covid']
            p_severe = sample['prediction']['severe']
            return p_covid, p_severe
