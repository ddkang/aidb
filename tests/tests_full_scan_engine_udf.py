import os
import time
import unittest

from collections import Counter
from unittest import IsolatedAsyncioTestCase

import pandas as pd
from sqlalchemy.sql import text

from aidb.utils.asyncio import asyncio_run
from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

from multiprocessing import Process

setup_test_logger('full_scan_engine')

DB_URL = "sqlite+aiosqlite://"


# DB_URL = "mysql+aiomysql://aidb:aidb@localhost"
class FullScanEngineTests(IsolatedAsyncioTestCase):

  def add_user_defined_function(self, aidb_engine):
    def sum_function(*args):
      return sum(args)


    def is_equal(col1, col2):
      return col1 == col2


    def max_function(*args):
      return max(args)


    def power_function(col1, col2):
      return col1**col2


    def replace_color(column1, selected_color, new_color):
      if column1 == selected_color:
        return new_color
      else:
        return column1


    async def async_objects_inference(blob_id):
      # Simulate some async operation
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'objects00':
          inference_service = service
          outputs = await inference_service.infer(blob_id)
          return outputs[0]


    def objects_inference(blob_id):
      df = pd.DataFrame({'blob_id': [blob_id]})
      return asyncio_run(async_objects_inference(df))


    async def async_lights_inference(blob_id):
      # Simulate some async operation
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'lights01':
          inference_service = service
          outputs = await inference_service.infer(blob_id)
          return outputs[0]


    def lights_inference(blob_id):
      df = pd.DataFrame({'blob_id': [blob_id]})
      return asyncio_run(async_lights_inference(df))


    async def async_colors_inference(blob_id):
      # Simulate some async operation
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'colors02':
          inference_service = service
          outputs = await inference_service.infer(blob_id)
          return outputs[0]


    def colors_inference(blob_id, object_id):
      df = pd.DataFrame({'blob_id': [blob_id], 'input_col2': object_id})
      return asyncio_run(async_colors_inference(df))


    async def async_counts_inference(blob_id):
      # Simulate some async operation
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'counts03':
          inference_service = service
          outputs = await inference_service.infer(blob_id)
          return outputs[0]


    def counts_inference(blob_id):
      df = pd.DataFrame({'blob_id': [blob_id]})
      return asyncio_run(async_counts_inference(df))


    aidb_engine._config.add_user_defined_function('sum_function', sum_function)
    aidb_engine._config.add_user_defined_function('is_equal', is_equal)
    aidb_engine._config.add_user_defined_function('max_function', max_function)
    aidb_engine._config.add_user_defined_function('power_function', power_function)
    aidb_engine._config.add_user_defined_function('replace_color', replace_color)
    aidb_engine._config.add_user_defined_function('objects_inference', objects_inference)
    aidb_engine._config.add_user_defined_function('lights_inference', lights_inference)
    aidb_engine._config.add_user_defined_function('colors_inference', colors_inference)
    aidb_engine._config.add_user_defined_function('counts_inference', counts_inference)



  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    self.add_user_defined_function(aidb_engine)

    queries = [
      # join condition test
      (
        'full_scan',
        '''
        SELECT y_max, x_min, y_min, color
        FROM objects00 join colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',
        '''
        SELECT y_max, x_min, y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      (
        'full_scan',
        '''
        SELECT y_max, x_min, y_min, color
        FROM objects00 join colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE AND is_equal(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800))
        ''',
        '''
        SELECT y_max, x_min, y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      # test user defined function in SELECT clause
      (
        'full_scan',
        '''
        SELECT max_function(x_min, y_min), y_min, replace_color(color, 'blue', 'new_blue')
        FROM objects00 join colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE AND is_equal(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800))
        ''',
        '''
        SELECT MAX(x_min, y_min), y_min, IIF(color='blue', 'new_blue', color)
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      # test comparison between user defined functions
      (
        'full_scan',
        '''
        SELECT y_min, color
        FROM objects00 join colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE max_function(x_min, x_max) < max_function(y_min, y_max) AND x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',
        '''
        SELECT y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE MAX(x_min, x_max) < MAX(y_min, y_max) AND  x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      # # test user defined function with constant parameters
      (
        'full_scan',
        '''
        SELECT max_function(y_max, y_min), power_function(x_min, 2), y_min, color
        FROM objects00 join colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE AND is_equal(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800))
        ''',
        '''
        SELECT MAX(y_max, y_min), POWER(x_min, 2), y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      # test user defined function in filter predicates
      (
        'full_scan',
        '''
        SELECT max_function(y_max, y_min), y_min, color
        FROM objects00 join colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE power_function(x_min, 2) > 640000 AND (x_min > 600 OR (x_max >600 AND y_min > 800))
        ''',
        '''
        SELECT MAX(y_max, y_min), y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE POWER(x_min, 2) > 640000 AND (x_min > 600 OR (x_max >600 AND y_min > 800))
        '''
      ),
      # named function for exact aggregation query
      (
        'full_scan',
        '''
        SELECT sum_function(SUM(x_min), SUM(y_max))
        FROM objects00
        ''',
        '''
        SELECT SUM(x_min) + SUM(y_max)
        FROM objects00
        '''
      ),
      # test machine learning model, output may be zero, multiple rows or multiple columns
      (
        'full_scan',
        '''
        SELECT objects_inference(frame)
        FROM blobs_00
        ''',
        '''
        SELECT object_name, confidence_score, x_min, y_min, x_max, y_max, object_id, frame
        FROM objects00
        '''
      ),
      (
        'full_scan',
        '''
        SELECT counts_inference(frame)
        FROM blobs_00
        ''',
        '''
        SELECT count, frame
        FROM counts03
        '''
      ),
      (
        'full_scan',
        '''
        SELECT lights_inference(frame)
        FROM blobs_00
        ''',
        '''
        SELECT light_1, light_2, light_3, light_4, frame
        FROM lights01
        '''
      ),
      # test user function with alias and filter based on the outputs of user function
      (
        'full_scan',
        '''
        SELECT objects_inference(frame) AS (output1, output2, output3, output4, output5, output6, output7, output8)
        FROM blobs_00
        WHERE (output3 > 600 AND output6 < 1400) OR frame < 1000
        ''',
        '''
        SELECT object_name, confidence, x_min, y_min, x_max, y_max, object_id, frame
        FROM objects00
        WHERE (x_min > 600 AND y_max < 1400) OR frame < 1000
        '''
      ),
      (
        'full_scan',
        '''
        SELECT lights_inference(frame) AS (output1, output2, output3, output4, output5)
        FROM blobs_00
        WHERE (output1 = 'red' AND output2 LIKE 'red') OR output3 = 'yellow'
        ''',
        '''
        SELECT light_1, light_2, light_3, light_4, frame
        FROM lights01
        WHERE (light_1 = 'red' AND light_2 LIKE 'red') OR light_3 = 'yellow'
        '''
      ),
      (
        'full_scan',
        '''
        SELECT colors_inference(frame, object_id) AS (output1, output2, output3), x_min, y_max
        FROM objects00
        WHERE (x_min > 600 AND output1 LIKE 'blue') OR (y_max < 1000 AND x_max < 1000)
        ''',
        '''
        SELECT color, colors02.frame, colors02.object_id, x_min, y_max
        FROM objects00 JOIN colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE (x_min > 600 AND color LIKE 'blue') OR (y_max < 1000 AND x_max < 1000)
        '''
      ),
      (
        'full_scan',
        '''
        SELECT colors_inference(frame, object_id) AS (output1, output2, output3), x_min AS col1, y_max AS col2
        FROM objects00
        WHERE (col1 > 600 AND output1 LIKE 'blue') OR (col2 < 1000 AND x_max < 1000)
        ''',
        '''
        SELECT color, colors02.frame, colors02.object_id, x_min AS col1, y_max AS col2
        FROM objects00 JOIN colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE (col1 > 600 AND color LIKE 'blue') OR (col2 < 1000 AND x_max < 1000)
        '''
      ),
      (
        'full_scan',
        '''
        SELECT max_function(y_max, y_min) AS output1, power_function(x_min, 2) AS output2, y_min, color
        FROM objects00 join colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE AND is_equal(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800)) AND output1 > 1000 AND output2 > 640000
        ''',
        '''
        SELECT MAX(y_max, y_min) AS output1, POWER(x_min, 2) AS output2, y_min, color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE (x_min > 600 OR (x_max >600 AND y_min > 800)) AND output1 > 1000 AND output2 > 640000
        '''
      ),
    ]

    for query_type, aidb_query, exact_query in queries:
      logger.info(f'Running query {exact_query} in ground truth database')
      # Run the query on the ground truth database
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(exact_query))
        gt_res = gt_res.fetchall()
      # Run the query on the aidb database
      logger.info(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      # TODO: may have problem with decimal number
      assert len(gt_res) == len(aidb_res)
      assert Counter(gt_res) == Counter(aidb_res)
    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()