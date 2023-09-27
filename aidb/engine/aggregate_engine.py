import pandas as pd
from typing import List
from sqlalchemy.sql import text
from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query


class AggregateEngine(BaseEngine):
	def get_tables(self, columns: List[str]) -> List[str]:
		tables = set()
		for col in columns:
			table_name = col.split('.')[0]
			tables.add(table_name)
		return list(tables)

	def get_join_str(self, tables: List[str]) -> str:
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

	def get_sampling_query(self, inp_cols_str, inp_tables, join_str, num_samples=100):
		random_sampler = f'''
						SELECT {inp_cols_str}
						FROM {', '.join(inp_tables)}
						{join_str}
						ORDER BY random()
						LIMIT {num_samples};
						'''
		return random_sampler

	async def execute_aggregate_query(self, query_str):
		# Need to use Query class in future, to get inp cols, tables concerned
		# instead of using functions like get_tables
		query = Query(query_str)
		# query.process_aggregate_query()
		service_ordering = self._config.inference_topological_order
		for bound_service in service_ordering:
			binding = bound_service.binding
			inp_cols = binding.input_columns

			inp_cols_str = ', '.join(inp_cols)
			inp_tables = self.get_tables(inp_cols)

			join_str = self.get_join_str(inp_tables)
			inp_query_str = self.get_sampling_query(inp_cols_str, inp_tables, join_str)

			async with self._sql_engine.begin() as conn:
				inp_df = await conn.run_sync(
								lambda conn: pd.read_sql(text(inp_query_str), conn)
							)
			await bound_service.infer(inp_df)

		async with self._sql_engine.begin() as conn:
			res = await conn.execute(text(query.sql))
			return res.fetchall()
		
