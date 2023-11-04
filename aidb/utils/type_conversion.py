def pandas_dtype_to_native_type(val):
  try:
    return val.item()
  except AttributeError:
    # in case of string or bytes
    return val