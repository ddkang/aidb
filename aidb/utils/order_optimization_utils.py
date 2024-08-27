from collections import defaultdict

from aidb.config.config import Config
from aidb.query.query import Query


def get_currently_supported_filtering_predicates_for_ordering(config: Config, query: Query):
  inference_engines_on_blob_ids = []
  all_columns_produced_by_blob_inference = {}
  for bound_inference_service in query.inference_engines_required_for_query:
    input_columns = bound_inference_service.binding.input_columns
    output_columns = bound_inference_service.binding.output_columns
    if all([input_column.split('.')[0] in config.blob_tables for input_column in input_columns]):
      inference_engines_on_blob_ids.append(bound_inference_service)
      for output_column in output_columns:
        all_columns_produced_by_blob_inference[output_column] = bound_inference_service
  supported_filtering_predicates = defaultdict(list)
  for connected_by_or_filtering_predicate in query.filtering_predicates:
    inference_engine_needed = set()
    for filtering_predicate in connected_by_or_filtering_predicate:
      for col in query.columns_in_filtering_predicate[filtering_predicate]:
        if col in all_columns_produced_by_blob_inference:
          inference_engine_needed.add(all_columns_produced_by_blob_inference[col])
    if len(inference_engine_needed) == 1:
      inference_engine = list(inference_engine_needed)[0]
      if inference_engine in inference_engines_on_blob_ids:
        supported_filtering_predicates[inference_engine].append(connected_by_or_filtering_predicate)
  return supported_filtering_predicates


def reorder_inference_engine(engine_to_proxy_score, static_order):
  engines_with_proxy_score_0 = []
  engine_scores = []
  for engine, score in engine_to_proxy_score.items():
    if engine.service.cost == 0:
      engines_with_proxy_score_0.append((engine, 0))
    elif score == 0:
      engines_with_proxy_score_0.append((engine, engine.service.cost))
    else:
      engine_scores.append((engine, engine.service.cost * score))
      
  engine_scores.sort(key=lambda a: a[1])
  engines_with_proxy_score_0.sort(key=lambda a: a[1])
  # TODO: Add support for cached engines
  ordered_engines = [e for e, s in engines_with_proxy_score_0] + [e for e, s in engine_scores]
  for e in static_order:
    if e not in ordered_engines:
      ordered_engines.append(e)
  return ordered_engines