
class Sampler():
  def __init__(self, mode='random'):
    self.mode = mode

  def get_random_sampling_query(self, inp_cols_str, inp_tables,
                                join_str, num_blob_ids, num_samples=100):
    random_sampler = f'''
                      SELECT {inp_cols_str}
                      FROM {', '.join(inp_tables)}
                      {join_str}
                      ORDER BY random()
                      LIMIT {num_samples};
                      '''
    scaling_factor = num_blob_ids // num_samples
    return random_sampler, scaling_factor

  def get_num_blob_ids_query(self, inp_col, inp_cols_str, inp_tables,
                            join_str, num_samples=100):
    num_ids_query = f'''
                      SELECT COUNT({inp_col})
                      FROM {', '.join(inp_tables)}
                      {join_str};
                      '''
    return num_ids_query