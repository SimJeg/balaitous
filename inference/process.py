from typing import Dict
from pathlib import Path
import SimpleITK

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device
from algorithm.balaitous import Balaitous


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        # load model
        self.model = Balaitous(device)

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        #try:
        sample = self.model(input_image)
        prob_covid = sample['prediction']['covid']
        prob_severe = sample['prediction']['severe']
        #except:
        #    print(f'Failed to process image')
        #    prob_covid = 0.60
        #    prob_severe = 0.25

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
