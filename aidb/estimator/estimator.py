import abc
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy
import scipy.stats
from statsmodels.stats.weightstats import DescrStatsW

from aidb.samplers.sampler import SampledBlob
from aidb.utils.logger import logger



@dataclass
class Estimate:
  estimate: float
  upper_bound: float
  lower_bound: float
  std: float
  std_ub: float


def _get_estimate_bennett(
    estimate: float,
    std: float,
    max_statistic: float,
    num_samples: int,
    conf: float
) -> Estimate:
  delta_inp = 1. - conf
  b = scipy.stats.chi2.ppf(delta_inp / 2, num_samples - 1)  # lower critical value
  std_ub = std * np.sqrt(num_samples - 1) / np.sqrt(b)
  var_ub = std_ub ** 2 * num_samples
  logger.info(f'estimate: {estimate}, std: {std}, max_statistic: {max_statistic},num_samples: {num_samples}, conf: {conf}')

  delta = delta_inp / 4.0
  a = max_statistic
  t = var_ub / a * \
      (np.exp(
        scipy.special.lambertw(
          (a ** 2 * np.log(1. / delta) - var_ub) \
          / (math.e * var_ub)
        ) + 1) - 1)
  t /= num_samples
  t = t.real

  lb = estimate - t
  ub = estimate + t

  ans = Estimate(estimate, ub, lb, std, std_ub)
  return ans


class Estimator(abc.ABC):
  def __init__(self, population_size: int) -> None:
    self._population_size = population_size


  @abc.abstractmethod
  def estimate(self, samples: List[SampledBlob], num_samples: int, conf: float, **kwargs) -> Estimate:
    pass


# Set estimators
class WeightedMeanSetEstimator(Estimator):
  def estimate(self, samples: List[SampledBlob], num_samples: int, conf: float, **kwargs) -> Estimate:
    weights = np.array([sample.weight for sample in samples])
    statistics = np.array([sample.statistic for sample in samples])
    counts = np.array([sample.num_items for sample in samples])
    cstats = np.repeat(statistics, counts)
    weights = np.repeat(weights, counts)
    wstats = DescrStatsW(cstats, weights=weights, ddof=0)
    return _get_estimate_bennett(
      wstats.mean,
      wstats.std,
      np.abs(cstats).max(),
      len(cstats),
      conf
    )


class WeightedCountSetEstimator(Estimator):
  def estimate(self, samples: List[SampledBlob], num_samples: int,  conf: float, **kwargs) -> Estimate:
    weight = samples[0].weight
    weights = np.array([weight] * num_samples)
    # Statistics are already counts

    statistics = np.array([sample.statistic for sample in samples] + [0] * (num_samples - len(samples)))
    wstats = DescrStatsW(statistics, weights=weights,  ddof=0)
    mean_est = _get_estimate_bennett(
      wstats.mean,
      wstats.std,
      np.abs(statistics).max(),
      len(statistics),
      conf
    )

    return Estimate(
      mean_est.estimate * self._population_size,
      mean_est.upper_bound,
      mean_est.lower_bound,
      mean_est.std,
      mean_est.std_ub
    )


# Logic is exactly the same for the count estimator
class WeightedSumSetEstimator(WeightedCountSetEstimator):
  pass