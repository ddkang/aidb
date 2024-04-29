from aidb.utils.logger import logger
from aidb.inference.bound_inference_service import CachedBoundInferenceService
from aidb.engine.engine import Engine
from sqlalchemy.sql import delete

async def clear_ML_cache(engine: Engine):
    '''
    Clear the cache table at start if the ML model has changed.
    Delete the cache table for each service.
    '''
    for service_binding in engine._config.inference_bindings:
        if isinstance(service_binding, CachedBoundInferenceService):
            async with service_binding._engine.begin() as conn:
                stmt = delete(service_binding._cache_table)
                await conn.execute(stmt)
        else:
            logger.debug(f"Service binding for {service_binding.service.name} is not cached")