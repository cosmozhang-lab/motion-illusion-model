class CommonConfig:
  def __init__(self, init = {}):
    self._dict = init
  def __getattr__(self, name):
    if name[0] == "_":
      return getattr(self, name)
    elif name in self._dict:
      return self._dict[name]
    else:
      return None
  def __setattr__(self, name, val):
    if name[0] == "_":
      self.__dict__[name] = val
    else:
      self._dict[name] = val
  def fetch(self, name, default = None):
    if name in self._dict:
      return self._dict[name]
    else:
      return default