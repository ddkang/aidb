from typing import List

import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: str, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    def get_tables(columns: List[str]) -> List[str]:
      tables = set()
      for col in columns:
        table_name = col.split('.')[0]
        tables.add(table_name)
      return list(tables)

    def get_join_str(tables: List[str]) -> str:
      table_pairs = [(tables[i], tables[i+1]) for i in range(len(tables) - 1)]
      join_strs = []
      for table1_name, table2_name in table_pairs:
        table1 = self._config.tables[table1_name]
        join_cols = []
        for fkey in table1.foreign_keys:
          if fkey.startswith(table2_name + '.'):
            join_cols.append((fkey, fkey.replace(table2_name, table1_name, 1)))
        join_strs.append(f'INNER JOIN {table2_name} ON {" AND ".join([f"{col1} = {col2}" for col1, col2 in join_cols])}')
      return '\n'.join(join_strs)


    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    for service, binding in service_ordering:
      inp_cols = binding.input_columns
      inp_cols_str = ', '.join(inp_cols)
      inp_tables = get_tables(inp_cols)
      join_str = get_join_str(inp_tables)
      inp_query_str = f'''
        SELECT {inp_cols_str}
        FROM {', '.join(inp_tables)}
        {join_str};
      '''

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      out_dfs = self.inference(inp_df, service)
      # We do not need to know the per-row information for a full scan
      out_df = pd.concat(out_dfs, ignore_index=True)
      out_df = self.process_inference_outputs(binding, out_df)
      out_cols = binding.output_columns
      out_tables = get_tables(out_cols)

      async with self._sql_engine.begin() as conn:
        out_df_cols = list(out_df.columns)
        for out_table in out_tables:
          insert_cols = [col for col in out_df_cols if col.startswith(out_table + '.')]
          insert_df = out_df[insert_cols]
          # At this point, the columns are fully qualified names (table_name.col_name). We need to remove the table name
          insert_df.columns = [col.split('.')[1] for col in insert_df.columns]
          await conn.run_sync(
            lambda conn: insert_df.to_sql(out_table, conn, if_exists='append', index=False)
          )

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query))
      return res.fetchall()