def get_inference_engines_required(filtering_predicates, columns_by_service, column_to_root_col):
  """
  Inference services required to run to satisfy the columns present in each filtering predicate
  for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
  it returns [[color], [], [object]]
  """
  inference_engines_required_predicates = []
  for filtering_predicate in filtering_predicates:
    inference_engines_required = set()
    for or_connected_predicate in filtering_predicate:
      if or_connected_predicate.left_exp.type == "column":
        originated_from = column_to_root_col.get(or_connected_predicate.left_exp.value,
                                                 or_connected_predicate.left_exp.value)
        if originated_from in columns_by_service:
          inference_engines_required.add(columns_by_service[originated_from].service.name)
      if or_connected_predicate.right_exp.type == "column":
        originated_from = column_to_root_col.get(or_connected_predicate.right_exp.value,
                                                 or_connected_predicate.right_exp.value)
        if originated_from in columns_by_service:
          inference_engines_required.add(columns_by_service[originated_from].service.name)
    inference_engines_required_predicates.append(inference_engines_required)
  return inference_engines_required_predicates


def get_tables_required(filtering_predicates, column_to_root_col):
  """
  Tables needed to satisfy the columns present in each filtering predicate
  for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
  it returns [[color], [blob], [object]]
  """
  tables_required_predicates = []
  for filtering_predicate in filtering_predicates:
    tables_required = set()
    for or_connected_predicate in filtering_predicate:
      if or_connected_predicate.left_exp.type == "column":
        originated_from = column_to_root_col.get(or_connected_predicate.left_exp.value,
                                                 or_connected_predicate.left_exp.value)
        tables_required.add(originated_from.split('.')[0])
      if or_connected_predicate.right_exp.type == "column":
        originated_from = column_to_root_col.get(or_connected_predicate.right_exp.value,
                                                 or_connected_predicate.right_exp.value)
        tables_required.add(originated_from.split('.')[0])
    tables_required_predicates.append(tables_required)
  return tables_required_predicates
