class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val  # noqa: E721

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
