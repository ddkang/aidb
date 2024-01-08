import os
import time
import unittest

from collections import Counter
from decimal import Decimal
from unittest import IsolatedAsyncioTestCase

import numpy as np
import pandas as pd
from sqlalchemy.sql import text

from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

from multiprocessing import Process

setup_test_logger('full_scan_engine_udf')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'


class FullScanEngineUdfTests(IsolatedAsyncioTestCase):

  def add_user_defined_function(self, aidb_engine):
    def sum_function(*args):
      return sum(args)


    def is_equal(col1, col2):
      return col1 == col2


    def max_function(*args):
      return max(args)


    def power_function(col1, col2):
      return col1**col2


    def multiply_function(col1, col2):
      return col1 * col2


    def replace_color(column1, selected_color, new_color):
      if column1 == selected_color:
        return new_color
      else:
        return column1


    async def async_objects_inference(blob_id):
      input_df = pd.DataFrame({'blob_id': [blob_id]})
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'objects00':
          inference_service = service
          outputs = await inference_service.infer(input_df)
          return outputs[0]


    async def async_lights_inference(blob_id):
      input_df = pd.DataFrame({'blob_id': [blob_id]})
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'lights01':
          inference_service = service
          outputs = await inference_service.infer(input_df)
          return outputs[0]


    async def async_colors_inference(blob_id, object_id):
      input_df = pd.DataFrame({'blob_id': [blob_id], 'input_col2': object_id})
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'colors02':
          inference_service = service
          outputs = await inference_service.infer(input_df)
          return outputs[0]


    async def async_counts_inference(blob_id):
      input_df = pd.DataFrame({'blob_id': [blob_id]})
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'counts03':
          inference_service = service
          outputs = await inference_service.infer(input_df)
          return outputs[0]


    aidb_engine._config.add_user_defined_function('sum_function', sum_function)
    aidb_engine._config.add_user_defined_function('is_equal', is_equal)
    aidb_engine._config.add_user_defined_function('max_function', max_function)
    aidb_engine._config.add_user_defined_function('multiply_function', multiply_function)
    aidb_engine._config.add_user_defined_function('power_function', power_function)
    aidb_engine._config.add_user_defined_function('replace_color', replace_color)
    aidb_engine._config.add_user_defined_function('objects_inference', async_objects_inference)
    aidb_engine._config.add_user_defined_function('lights_inference', async_lights_inference)
    aidb_engine._config.add_user_defined_function('colors_inference', async_colors_inference)
    aidb_engine._config.add_user_defined_function('counts_inference', async_counts_inference)



  async def test_udf_sqlite(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    port = 8010
    p = Process(target=run_server, args=[str(data_dir), port])
    p.start()
    time.sleep(1)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(SQLITE_URL, data_dir, port)

    register_inference_services(aidb_engine, data_dir, port)

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
      # test user defined function with constant parameters
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
        SELECT object_name, confidence_score, x_min, y_min, x_max, y_max, object_id, frame
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
      try:
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
      finally:
        await gt_engine.dispose()
      # Run the query on the aidb database
      logger.info(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      assert len(gt_res) == len(aidb_res)
      assert Counter(gt_res) == Counter(aidb_res)
      assert sorted(gt_res) == sorted(aidb_res)
    del gt_engine
    del aidb_engine
    p.terminate()


  async def test_udf_postgresql_and_mysql(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)

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
      # test machine learning model, output may be zero, multiple rows or multiple columns
      (
        'full_scan',
        '''
        SELECT objects_inference(frame)
        FROM blobs_00
        WHERE frame = 300
        ''',
        '''
        SELECT object_name, confidence_score, x_min, y_min, x_max, y_max, object_id, frame
        FROM objects00
        WHERE frame = 300
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
        SELECT object_name, confidence_score, x_min, y_min, x_max, y_max, object_id, frame
        FROM objects00
        WHERE (x_min > 600 AND y_max < 1400) OR frame < 1000
        '''
      ),
      # test UDFs created within the database and AIDB
      (
        'full_scan',
        '''
        SELECT multiply_function(x_min, y_min), database_multiply_function(x_min, y_min), x_max, y_max
        FROM objects00
        WHERE x_min > 600 AND y_max < 1000
        ''',
        '''
        SELECT database_multiply_function(x_min, y_min), database_multiply_function(x_min, y_min), x_max, y_max
        FROM objects00
        WHERE x_min > 600 AND y_max < 1000
        '''
      ),
      (
        'full_scan',
        '''
        SELECT database_add_function(y_max, x_min), multiply_function(y_min, y_max), color
        FROM objects00 join colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',
        '''
        SELECT database_add_function(y_max, x_min), database_multiply_function(y_min, y_max), color
        FROM objects00 join colors02 ON objects00.frame = colors02.frame AND objects00.object_id = colors02.object_id
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        '''
      ),
      (
        'full_scan',
        '''
        SELECT frame, database_multiply_function(x_min, y_min), sum_function(x_max, y_max)
        FROM objects00
        WHERE (multiply_function(x_min, y_min) > 400000 AND database_add_function(y_max, x_min) < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        ''',
        '''
        SELECT frame, database_multiply_function(x_min, y_min), database_add_function(x_max, y_max)
        FROM objects00
        WHERE (database_multiply_function(x_min, y_min) > 400000 AND database_add_function(y_max, x_min) < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        '''
      ),
      (
        'full_scan',
        '''
        SELECT frame, database_multiply_function(x_min, y_min), sum_function(x_max, y_max) AS output1
        FROM objects00
        WHERE (multiply_function(x_min, y_min) > 400000 AND output1 < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        ''',
        '''
        SELECT frame, database_multiply_function(x_min, y_min), database_add_function(x_max, y_max)
        FROM objects00
        WHERE (database_multiply_function(x_min, y_min) > 400000 AND database_add_function(x_max, y_max) < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        '''
      ),
    ]

    postgresql_function =  [
        '''
        CREATE OR REPLACE FUNCTION database_multiply_function(col1 DOUBLE PRECISION,
            col2 DOUBLE PRECISION)
        RETURNS double precision AS
        $$
        BEGIN
          RETURN (col1 * col2)::double precision;
        END;
        $$
        LANGUAGE plpgsql;
        ''',
        '''
        CREATE OR REPLACE FUNCTION database_add_function(col1 DOUBLE PRECISION,
            col2 DOUBLE PRECISION)
        RETURNS double precision AS
        $$
        BEGIN
          RETURN (col1 + col2)::double precision;
        END;
        $$
        LANGUAGE plpgsql;
        '''
    ]

    mysql_function = [
        '''
        CREATE FUNCTION database_multiply_function(col1 FLOAT(32), col2 FLOAT(32)) 
        RETURNS FLOAT(32) 
        DETERMINISTIC 
        BEGIN DECLARE multiply_result FLOAT(32); SET multiply_result = col1 * col2; 
        RETURN multiply_result; 
        END 
        ''',
        '''
        CREATE FUNCTION database_add_function(col1 FLOAT(32), col2 FLOAT(32))
        RETURNS FLOAT(32)
        DETERMINISTIC
        BEGIN
            DECLARE add_result FLOAT(32);
            SET add_result = col1 + col2;
            RETURN add_result;
        END
        '''
    ]
    for db_url in [POSTGRESQL_URL, MYSQL_URL]:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      if dialect == 'mysql':
        function_list = mysql_function
      elif dialect == 'postgresql':
        function_list = postgresql_function
      else:
        raise Exception('Unsupported database')

      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir)

      try:
        async with gt_engine.begin() as conn:
          await conn.execute(text('DROP FUNCTION IF EXISTS database_multiply_function;'))
          await conn.execute(text('DROP FUNCTION IF EXISTS database_add_function;'))
          for function in function_list:
            await conn.execute(text(function))
      finally:
        await gt_engine.dispose()

      # FIXME(ttt-77): create function SQL can't be executed by AIDB engine
      try:
        async with aidb_engine._sql_engine.begin() as conn:
          await conn.execute(text('DROP FUNCTION IF EXISTS database_multiply_function;'))
          await conn.execute(text('DROP FUNCTION IF EXISTS database_add_function;'))
          for function in function_list:
            await conn.execute(text(function))
      finally:
        await aidb_engine._sql_engine.dispose()


      register_inference_services(aidb_engine, data_dir)
      self.add_user_defined_function(aidb_engine)

      for query_type, aidb_query, exact_query in queries:
        logger.info(f'Running query {exact_query} in ground truth database')
        # Run the query on the ground truth database
        try:
          async with gt_engine.begin() as conn:
            gt_res = await conn.execute(text(exact_query))
            gt_res = gt_res.fetchall()
        finally:
          await gt_engine.dispose()
        # Run the query on the aidb database
        logger.info(f'Running query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)

        gt_res = sorted(gt_res)
        aidb_res = sorted(aidb_res)

        relative_diffs = []
        for i in range(len(gt_res)):
          relative_diff = []
          for element1, element2 in zip(gt_res[i], aidb_res[i]):
            if isinstance(element1, (int, float, Decimal)) and (element2 != 0 or element1 !=0):
              x = (2 * abs(element1 - element2)) / (abs(element1) + abs(element2)) * 100
              assert x <= 0.0001
              relative_diff.append(x)
            else:
              assert element2 == element1
              relative_diff.append(0)
          relative_diffs.append(relative_diff)
        avg_diff = np.mean(np.mean(relative_diffs, axis=1))
        print(f'Avg relative difference percentage between gt_res and aidb_res: {avg_diff}')

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
