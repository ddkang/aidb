from typing import Dict, List

import numpy as np
import pandas as pd
import sqlglot.expressions as exp
from sqlalchemy.sql import text

from aidb.engine.approx_aggregate_engine import ApproximateAggregateEngine
from aidb.query.query import Query
from aidb.utils.constants import (MASS_COL_NAME, NUM_ITEMS_COL_NAME,
                                  WEIGHT_COL_NAME)
from aidb.utils.logger import logger

_NUM_PILOT_SAMPLES = 60000

RETRIEVAL_BATCH_SIZE = 20000

class ApproximateAggregateJoinEngine(ApproximateAggregateEngine):
  async def get_rows_in_join_tables(self, blob_tables: List[str], blob_key_filtering_predicates):
    '''
    Function to return a list of dataframes, with each DataFrame containing all rows from a corresponding blob table
    '''
    sample_df_list = []

    for table in blob_tables:
      cols = [col.name for col in self._config.tables[table].columns]
      fp_list = []
      for fp in blob_key_filtering_predicates:
        if len(fp) > 1:
          raise Exception('OR operator for blob keys filtering is not supported.')
        if fp[0].find(exp.Column).args['table'].args['this'] == table:
          fp_list.append(fp)

      where_str = self._get_where_str(fp_list)
      where_str = f'WHERE {where_str}' if where_str else ''
      select_str = ', '.join(cols)
      query_str = f'''
                   SELECT {select_str}
                   FROM {table}
                   {where_str};
                   '''
      async with self._sql_engine.begin() as conn:
        sample_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(query_str), conn))
        sample_df_list.append(sample_df)

    return sample_df_list


  def execute_join_query_sampling(self, sample_df_list, num_samples, col_to_alias, agg_col_to_alias):
    data_size = (len(sample_df_list[0]), len(sample_df_list[1]))

    self.blob_count = np.prod(data_size)
    sample = np.random.choice(self.blob_count, size=num_samples, replace=True)
    sample_ids = np.array(np.unravel_index(sample, data_size))

    sampled_cols = pd.concat(
        [sample_df_list[0].iloc[sample_ids[0]].reset_index(drop=True),
        sample_df_list[1].iloc[sample_ids[1]].reset_index(drop=True)],
        axis=1
    )

    sample_res_df = sampled_cols[list(col_to_alias.keys()) + list(agg_col_to_alias.keys())]
    sample_res_df.columns = list(col_to_alias.values()) + list(agg_col_to_alias.values())
    sample_res_df = sample_res_df.copy()
    sample_res_df[WEIGHT_COL_NAME] = 1. / self.blob_count
    sample_res_df[MASS_COL_NAME] = 1

    return sample_res_df


  async def execute_inference_services_join(
      self,
      query: Query,
      sample_df: pd.DataFrame,
      blob_keys: List[str],
      agg_list: List[tuple]
  ) -> List[pd.DataFrame]:
    '''
    Executed inference services on sampled data
    '''
    dataframe_sql, query = query.udf_query

    agg_cols = [agg_col for _, agg_col in agg_list]
    fixed_cols = [WEIGHT_COL_NAME, MASS_COL_NAME]
    res_satisfy_udf = self.execute_user_defined_function(
        sample_df,
        dataframe_sql,
        query,
        blob_keys + fixed_cols + agg_cols
    )

    # group by primary keys
    grouped_data = res_satisfy_udf.groupby(by=blob_keys + fixed_cols)

    agg_res_list = []

    agg_type_mapping = {exp.Count: 'count', exp.Sum: 'sum', exp.Avg: 'mean'}
    for agg_type, agg_col in agg_list:
      group_by_col_to_type = {}
      if agg_col == '*':
        agg_res = grouped_data.count().iloc[:,:2]
      else:
        group_by_col_to_type[agg_col] = [agg_type_mapping[agg_type], 'count']
        agg_res = grouped_data.agg(group_by_col_to_type)

      agg_res.columns = [agg_col, NUM_ITEMS_COL_NAME]
      agg_res_list.append(agg_res.reset_index())
    return agg_res_list


  async def execute_sampling_and_join_services(
      self,
      query: Query,
      blob_tables: List[str],
      num_samples: int,
      col_to_alias: Dict[str, str],
      agg_col_to_alias: Dict[str, str],
      agg_list: List[tuple]
  ) -> List[pd.DataFrame]:
    _, udf_query = query.udf_query

    # Extract filtering predicates for blob table queries
    filtering_predicates = udf_query.filtering_predicates
    blob_key_filtering_predicates = []
    inference_engines_required = udf_query.inference_engines_required_for_filtering_predicates

    for filtering_predicate, engine_required in zip(filtering_predicates, inference_engines_required):
      if len(engine_required) == 0:
        blob_key_filtering_predicates.append(filtering_predicate)

    sample_df_list = await self.get_rows_in_join_tables(blob_tables, blob_key_filtering_predicates)

    blob_keys = []
    for blob_table in blob_tables:
      for blob_key in self._config.tables[blob_table].primary_key:
        if blob_key in col_to_alias:
          blob_key = col_to_alias[blob_key]
        blob_keys.append(blob_key)

    res_store_list = []

    # selectively sample data to avoid large memory usage
    total_sampled_count = 0
    while total_sampled_count < num_samples:
      sample_res_df = self.execute_join_query_sampling(
          sample_df_list,
          RETRIEVAL_BATCH_SIZE,
          col_to_alias,
          agg_col_to_alias
      )
      if total_sampled_count + len(sample_res_df) > num_samples:
        sample_res_df = sample_res_df.iloc[:num_samples-total_sampled_count]
      total_sampled_count += len(sample_res_df)

      agg_res_list = await self.execute_inference_services_join(query, sample_res_df, blob_keys, agg_list)
      if len(agg_res_list) != 0:
        if len(res_store_list) == 0:
          res_store_list = agg_res_list
        else:
          res_store_list = [pd.concat([df_a, df_b], ignore_index=True)
                            for df_a, df_b in zip(res_store_list, agg_res_list)]

    return res_store_list


  def get_additional_required_num_samples_join(
      self,
      query: Query,
      sample_results: List[pd.DataFrame],
      agg_type_with_cols,
      alpha
  ) -> int:
    error_target = query.error_target
    num_samples = []

    index = 0
    while index < len(agg_type_with_cols):
      agg_type, agg_col = agg_type_with_cols[index]
      sample_result = sample_results[index]
      extracted_sample_results = sample_result[[agg_col, NUM_ITEMS_COL_NAME, WEIGHT_COL_NAME, MASS_COL_NAME]]
      num_samples.append(
        self._calculate_required_num_samples(
          extracted_sample_results,
          _NUM_PILOT_SAMPLES,
          agg_type,
          alpha,
          error_target
        )
      )
      index += 1

    return max(max(num_samples), 100)


  async def execute_aggregate_join_query(self, query: Query):
    '''
    Execute aggregation query using approximate processing
    '''
    query.is_valid_aqp_query()

    query_no_aqp = query.base_sql_no_aqp
    agg_type_with_cols = query_no_aqp.query_after_normalizing.aggregation_type_list_in_query

    dataframe_sql, udf_query = query_no_aqp.udf_query

    _, alias_to_col = udf_query.table_and_column_aliases_in_query

    agg_col_to_alias = {}
    agg_list = []

    for i, (agg_type, agg_col) in enumerate(agg_type_with_cols):
      if agg_col == '*':
        agg_list.append((agg_type, agg_col))
        continue
      if agg_col.split('.')[1] not in agg_col_to_alias:
        agg_col_to_alias[agg_col.split('.')[1]] = f'agg_col__{i}'
        udf_query = udf_query.add_select(f'{agg_col} AS agg_col__{i}')
      agg_list.append((agg_type, agg_col_to_alias[agg_col.split('.')[1]]))

    col_to_alias = {value: key for key, value in alias_to_col.items()}

    # FIXME(ttt-77): retrieve join tables to support self-join
    blob_tables = query_no_aqp.blob_tables_required_for_query
    if len(blob_tables) != 2:
      raise Exception('JOIN query require two blob tables')

    filtering_predicates = udf_query.filtering_predicates
    blob_key_filtering_predicates = []
    inference_engines_required = udf_query.inference_engines_required_for_filtering_predicates

    for filtering_predicate, engine_required in zip(filtering_predicates, inference_engines_required):
      if len(engine_required) == 0:
        blob_key_filtering_predicates.append(filtering_predicate)

    # Perform sampling and then conduct the inference process
    res_list = await self.execute_sampling_and_join_services(
        query_no_aqp,
        blob_tables,
        _NUM_PILOT_SAMPLES,
        col_to_alias,
        agg_col_to_alias,
        agg_list
    )

    for res in res_list:
      if len(res) == 0:
        raise Exception('No matched pair found. Please use a larger oracle budget.')

    num_aggregations = len(agg_list)
    alpha = (1. - (query.confidence / 100.)) / num_aggregations
    conf = 1. - alpha
    num_samples = self.get_additional_required_num_samples_join(query, res_list, agg_list, alpha)

    logger.info(f'num_samples: {num_samples}')

    new_res_list = await self.execute_sampling_and_join_services(
        query_no_aqp,
        blob_tables,
        num_samples,
        col_to_alias,
        agg_col_to_alias,
        agg_list
    )

    concatenated_results = []
    if len(new_res_list) != 0:
      for res, new_res in zip(res_list, new_res_list):
        concatenated_results.append(pd.concat([res, new_res], ignore_index=True))
    else:
      concatenated_results = res_list

    estimates = []
    index = 0
    while index < len(agg_type_with_cols):
      res_i = concatenated_results[index]
      agg_type, agg_col = agg_list[index]

      estimator = self._get_estimator(agg_type)
      sample_res = res_i[[agg_col, NUM_ITEMS_COL_NAME, WEIGHT_COL_NAME, MASS_COL_NAME]]
      estimates.append(
        estimator.estimate(
          sample_res,
          _NUM_PILOT_SAMPLES + num_samples,
          conf
        ).estimate
      )
      index += 1
    return [tuple(estimates)]
