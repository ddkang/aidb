def call_counter(func):
  def wrapper(*args, **kwargs):
    wrapper.calls += 1
    print(wrapper.calls)
    return func(*args, **kwargs)

  wrapper.calls = 0
  return wrapper
