from collections import deque

import networkx as nx
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import delete

from aidb.engine.engine import Engine
from aidb.inference.bound_inference_service import CachedBoundInferenceService
from aidb.utils.logger import logger


async def clear_ML_cache(engine: Engine):
  '''
  Clear the cache table at start if the ML model has changed.
  Delete the cache table for each service.
  '''
  for inference_binding in engine._config.inference_bindings:
    if isinstance(inference_binding, CachedBoundInferenceService):
      async with inference_binding._engine.begin() as conn:
        tables_to_delete = inference_binding.get_tables(inference_binding.binding.output_columns) + [inference_binding._cache_table_name]
        fk_ref_graph = nx.DiGraph()
        fk_ref_graph.add_nodes_from(tables_to_delete)
        for table_name in tables_to_delete:
          if table_name == inference_binding._cache_table_name:
            table = inference_binding._cache_table
          else:
            table = inference_binding._tables[table_name]._table
          for constraint in table.constraints:
            if isinstance(constraint, ForeignKeyConstraint):
              for fk in constraint.elements:
                fk_ref_table_name = fk.column.table.name
                if fk_ref_table_name in fk_ref_graph:
                  fk_ref_graph.add_edge(table_name,fk_ref_table_name)
        
        table_order = nx.topological_sort(fk_ref_graph)
        for table_name in table_order:
          if table_name == inference_binding._cache_table_name:
            table = inference_binding._cache_table
          else:
            table = inference_binding._tables[table_name]._table
          await conn.execute(delete(table))
        
    else:
      logger.debug(f"Service binding for {inference_binding.service.name} is not cached")