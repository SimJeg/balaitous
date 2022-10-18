# Balaitous


[![PyPI version](https://badge.fury.io/py/balaitous.svg)](https://badge.fury.io/py/balaitous)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/542006397.svg)](https://zenodo.org/badge/latestdoi/542006397)

Balaitous is an updated version of the AI-severity model described in [Lassau et al., 2021](https://doi.org/10.1038/s41467-020-20657-4).

Given an input CT scan, Balaitous outputs a probability for COVID disease and a probability for severe outcome, defined as intubation or death within one month.


## News 🚀

- October 2022 - The model trained on the private [STOIC database](https://pubs.rsna.org/doi/10.1148/radiol.2021210384) (n=9724) ranked 2nd 🥈 for severity prediction (AUC=81.0% vs 81.5% for 1st place) and 1st 🥇 for COVID diagnosis (AUC=84.5%) on the [final leaderboard](https://stoic2021.grand-challenge.org/evaluation/challenge-2/leaderboard/) (n=1000). Slides from the STOIC webinar can be found in the `assets` directory.
- September 2022 - The model trained on the public STOIC database is released (v1.0)
- April 2022 - The model trained on the public STOIC database (n=2000) ranked 1st 🥇 for severity prediction (AUC=80.4%) and 1st 🥇 for COVID diagnosis (AUC=83.2%) on the [qualification leaderboard](https://stoic2021.grand-challenge.org/evaluation/quallification-last-submission/leaderboard/) (n=800).

## Installation

```bash
pip install balaitous
```

## Usage

Using the command line interface:
```bash
balaitous run path/to/image
````

or using python (recommanded for batch predictions): 
```python
from balaitous import Balaitous

model = Balaitous()
p_covid, p_severe = model('path/to/image')
```

The input image must be readable using the `SimpleITK.ReadImage` function (*e.g.* .nii or .mha file). If your input is a DICOM folder, you can convert it using tools such as [dcm2niix](https://github.com/rordenlab/dcm2niix).

 `PatientAge` and `PatientSex` metadata keys are automatically parsed from the input image. If not available, age (in years, *e.g.* 65) and sex (1 for male, 0 for female) can be optionnaly passed to Balaitous :

```bash
balaitous run /path/to/image --age age --sex sex
```

or:
```python
p_covid, p_severe = model('path/to/image', age=age, sex=sex)
```

*Note: Balaitous runs much faster on GPU : 2-4 sec/sample on a GPU (NVIDIA GTX 1080Ti) compared to 2-4 min/sample on CPU (Intel i7, 8 cores).*

## Model description

The processing steps of Balaitous (see `balaitous.py`) are the following : 

- The scan is resized to a pixel spacing of (1.5mm, 1.5mm, 5mm) and reshaped to a shape of (224, 224, Z)
- A lung segmentation mask is obtained using a 2D U-Net ([source](https://github.com/JoHof/lungmask))
- The scan is cropped to the slices containing the lungs
- A first feature extractor is applied to get a first vector $X_{full}$
- The lung mask is applied to the image (only lungs are now visible)
- A second feature extractor is applied to get a second vector $X_{lung}$
- For the severe outcome, 2 logistic regressions are applied to [$X_{full}$, age, sex] and [$X_{lung}$, age, sex] and the 2 probabilities are averaged 
- For the covid outcome, 2 logistic regressions are applied to $X_{full}$ and $X_{lung}$ and the 2 probabilities are averaged 

The first feature extractor is a ViT-L model pretrained on ImageNet-22k using iBOT ([source](https://github.com/bytedance/ibot)) and finetuned for 35 epochs on 165k CT slices (4k patients from 7 public datasets). The second feature extractor is the same ViT-L model without finetuning. Model weights can be found on [Zenodo](https://zenodo.org/record/6547999#.Yn9QjJNBxSA).

Only the 4 logistic regressions were trained on the STOIC database, and only COVID positive patients were used to train the 2 logistic regressions for the prediction of severity. 

*Note : hyper-parameters and feature extractors have been choosen following cross-validation results on the public STOIC database (n=2000 patients). Using the finetuned iBOT model on the plain image instead of the ImageNet model only brought modest performance gains.* 

It is possible to get intermediate output variables from Balaitous using : 

```python
output_dict = model('path/to/image', return_dict=True)
```

The main keys of this dictionnary are : 
- `unmasked_image`: array of the resized image with shape (224, 224, Z) and (1.5mm, 1.5mm, 5mm) pixel spacing
- `mask`: boolean array of the lung mask 
- `image`: image with the lung mask applied
- `features`: dictionnary of features from the unmasked image (key `full`) and from the masked image (key `lung`)
- `prediction`: dictionnary of predictions for the 4 logistic regressions (keys `full_covid`, `lung_covid`, `full_severe`, `lung_severe`) and their weighted averages (keys `covid` and `severe`)

*Note: Balaitous predictions are invariant to vertical image flips (see the iBOTExtractor class in `transforms.py`). 
Such flips may happen depending on the DICOM conversion tools, so don't worry if `unmasked_image` and `image`  are flipped.*

## Performances

The ROC-AUC performances (in %) of Balaitous are:

|                       | AUC severity  | AUC covid      | 
| --                    | --            | --             | 
| Training - $X_{full}$  | 79.01 +- 2.63 | 80.65 +- 2.16  |
| Training - $X_{lung}$ | 79.00 +- 3.30 | 82.63 +- 1.99  |
| Training              | 80.36 +- 2.80 | 82.98 +- 2.01  |
| Qualification LB      | 80.44         | 83.22          |  
| Final LB              | 79.4°         | -          |  

There were n=2000 patients in the training dataset (n=1205 COVID positive), n=800 patients in the Qualification LB dataset, and n=1000 patients in the Final LB dataset.

 Performances on the training dataset are computed using a stratified 4x8-fold cross-validation scheme. Following the STOIC-2021 challenge, the AUC for the severity prediction task is computed only among COVID positive patients. 

° performance reported by the organizers during the STOIC webinar

## Calibration

Calibration has not been performed as the validation set has not been released.

## Interpretability

For the severity prediction task, most of the false negatives are patients with low lung lesion burden but a severe outcome. On the opposite, most of the false positives are patients with high lung lesion burden but a positive outcome. This tends to indicate that the lung lesion burden is the main feature used by the model.

More investigation should be done to interpret Balaitous inner working. For instance, the $X_{full}$ features contain information that are external to the lung. Some can be predictive of a severe outcome such as patient body composition or heart condition, but they may also contain biases such as the presence of catheters if the patient was intubated before getting its CT scan. 

## License

This repository has been released under the MIT license.

## Medical disclaimer

This repository is for the purpose of disseminating health information free of charge for the benefit of the public and research-sharing purposes only and is made available on the basis that no professional advice on a particular matter is being provided. Nothing contained in this repository is intended to be used as medical advice and it is not intended to be used to diagnose, treat, cure or prevent any disease, nor should it be used for therapeutic purposes or as a substitute for your own health professional’s advice. No liability is accepted for any injury, loss or damage incurred by use of or reliance on the information provided on this repository.


