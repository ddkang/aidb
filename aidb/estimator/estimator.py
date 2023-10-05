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
  b = scipy.stats.chi2.ppf(delta_inp / 2, num_samples - 1)
  std_ub = std * np.sqrt(num_samples - 1) / np.sqrt(b)
  var_ub = std_ub ** 2 * num_samples
  logger.info(f'estimate: {estimate}, std: {std}, max_statistic: {max_statistic}, num_samples: {num_samples}, conf: {conf}')

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

  return Estimate(estimate, ub, lb, std, std_ub)

class Estimator(abc.ABC):
  def __init__(self, population_size: int) -> None:
    self._population_size = population_size

  @abc.abstractmethod
  def estimate(self, samples: List[SampledBlob], num_samples: int, conf: float, **kwargs) -> Estimate:
    pass

# Single estimators
class WeightedMeanSingleEstimator(Estimator):
  def __init__(self, population_size: int) -> None:
    self._population_size = population_size

  def estimate(self, samples: List[SampledBlob], num_samples: int, conf: float, **kwargs) -> Estimate:
    weights = np.array([sample.weight for sample in samples])
    statistics = np.array([sample.statistic for sample in samples])
    wstats = DescrStatsW(statistics, weights=weights, ddof=0)
    return _get_estimate_bennett(
      wstats.mean,
      wstats.std,
      np.abs(statistics).max(),
      len(statistics),
      conf
    )


# Set estimators
class WeightedMeanSetEstimator(Estimator):
  def estimate(self, samples: List[SampledBlob], num_sampled: int, conf: float, **kwargs) -> Estimate:
    weights = np.array([sample.weight for sample in samples])
    print(f'weights: {weights}')
    statistics = np.array([sample.statistic for sample in samples])
    print(f'statistics: {statistics}')
    agg_table = kwargs.get("agg_table")
    print(f'agg_table: {agg_table}')
    counts = np.array([sample.num_items[agg_table] if agg_table in sample.num_items else 0 for sample in samples])
    print(f'counts: {counts}')
    cstats = np.repeat(statistics, counts)
    print(f'cstats: {cstats}')
    weights = np.repeat(weights, counts)
    print(f'weights modified: {weights}')
    wstats = DescrStatsW(cstats, weights=weights, ddof=0)
    print(f'wstats: {wstats}')
    print(f'max cstats: {np.abs(cstats).max()}')
    print(f'len cstats: {len(cstats)}')
    print(f'wstats.mean: {wstats.mean}')
    print(f'wstats.std: {wstats.std}')
    return _get_estimate_bennett(
      wstats.mean,
      wstats.std,
      np.abs(cstats).max(),
      len(cstats),
      conf
    )

class WeightedCountSetEstimator(WeightedMeanSingleEstimator):
  def estimate(self, samples: List[SampledBlob], num_sampled: int, conf: float, **kwargs) -> Estimate:
    weights = np.array([sample.weight for sample in samples])
    # Statistics are already counts
    statistics = np.array([sample.statistic for sample in samples])

    wstats = DescrStatsW(statistics, weights=weights, ddof=0)
    mean_est = _get_estimate_bennett(
      wstats.mean,
      wstats.std,
      np.abs(statistics).max(),
      len(statistics),
      conf
    )
    inflation_factor = (num_sampled / len(samples)) * self._population_size
    return Estimate(
      mean_est.estimate * inflation_factor,
      mean_est.upper_bound * inflation_factor,
      mean_est.lower_bound * inflation_factor,
      mean_est.std,
      mean_est.std_ub
    )

# Logic is exactly the same for the count estimator
class WeightedSumSetEstimator(WeightedCountSetEstimator):
  pass