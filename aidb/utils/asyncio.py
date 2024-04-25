import asyncio

import nest_asyncio


def asyncio_run(future, as_task=True):
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:  # no event loop running
    loop = asyncio.new_event_loop()
    ret = loop.run_until_complete(_to_task(future, as_task, loop))
    loop.close()
    return ret
  else:
    nest_asyncio.apply(loop)
    return asyncio.run(_to_task(future, as_task, loop))
  
def _to_task(future, as_task, loop):
  if not as_task or isinstance(future, asyncio.Task):
    return future
  return loop.create_task(future)


