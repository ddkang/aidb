import sqlglot.expressions as exp


class ColumnExtractor(object):
  def __init__(self):
    pass

  def _validate(self, node):
    for child in node.walk():
      if isinstance(child, exp.Lateral):
        raise Exception('Lateral not supported')
      if isinstance(child, exp.Join):
        raise Exception('Join not supported')
    return True


  def _extract_table(self, node):
    if not isinstance(node, exp.From):
      raise Exception('Table expressions can only be in FROM clauses')

    # Since joins aren't supported at the moment, only one possible table expression
    # There can be multiple aliases, which are ignored
    tables = []
    for _, v in node.args.items():
      child_nodes = v if isinstance(v, list) else [v]
      for cn in child_nodes:
        if isinstance(cn, exp.Table):
          tables.append(cn.args['this'].args['this'])

    if len(tables) > 1:
      raise Exception('Multiple tables not supported')
    if len(tables) == 0: return None
    return tables[0]


  def _extract_from_select(self, node):
    if not isinstance(node, exp.Select):
      raise Exception('Not a select expression')

    columns = []
    table = None
    table_columns = []
    for _, v in node.args.items():
      child_nodes = v if isinstance(v, list) else [v]

      for cn in child_nodes:
        if isinstance(cn, exp.From):
          table = self._extract_table(cn)
          table_columns.extend(self._extract(cn))
        elif isinstance(cn, exp.Expression):
          table_columns.extend(self._extract(cn))

    # TODO: these need to be correctly parsed and formated to a canonical form
    for column in table_columns:
      # Table name already in the column
      if '.' in column:
        columns.append(column)
      elif table is not None:
        columns.append(table + '.' + column)

    return columns


  def _extract(self, node):
    if not isinstance(node, exp.Expression):
      return []
    if isinstance(node, exp.Column):
      if isinstance(node.args['this'], exp.Identifier):
        return [node.args['this'].args['this']]
      elif isinstance(node.args['this'], exp.Star):
        # TODO: this requires the table metadata to extract the columns
        raise Exception('Star not supported')
        # return [node.args['this'].args['this']]
      else:
        raise Exception('Unsupported column type')

    if isinstance(node, exp.Select):
      return self._extract_from_select(node)

    columns = []
    for _, v in node.args.items():
      child_nodes = v if isinstance(v, list) else [v]
      for cn in child_nodes:
        columns.extend(self._extract(cn))
    return columns

  def extract(self, expression: exp.Expression):
    columns = self._extract(expression)
    columns = list(set(columns))
    return columns