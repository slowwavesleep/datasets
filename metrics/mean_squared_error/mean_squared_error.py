# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mean squared error metric"""
from typing import Dict

from sklearn.metrics import mean_squared_error

import datasets


_DESCRIPTION = """\
...
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    sample_weight: Sample weights.
    multioutput: Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        ‘raw_values’ :
        Returns a full set of errors in case of multioutput input.
        ‘uniform_average’ :
        Errors of all outputs are averaged with uniform weight.
    squared: If True returns MSE value, if False returns RMSE value.
Returns:
    mean_squared_error: A non-negative floating point value (the best value is 0.0), or an array of floating point
                        values, one for each individual target.
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MeanSquaredError(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"
            ],
        )

    def _compute(
        self, predictions, references, sample_weight=None, multioutput="uniform_average", squared=True
    ) -> Dict[str, float]:
        return {
            "mean_squared_error": float(
                mean_squared_error(
                    references, predictions, sample_weight=sample_weight, multioutput=multioutput, squared=squared
                )
            ),
        }
