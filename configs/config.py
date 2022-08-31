from logging import raiseExceptions
import os
import yaml

class DotDict(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __getitem__(self, k):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and '.' in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default=default)

class Config(DotDict):
    def __init__(self, fp=None, dic=None):
        if not fp and not dic:
            assert False, 'either one of config file or config dic should be exist'
        
        if fp:
            with open(fp)as file:
                super().__init__(yaml.safe_load(file))
        elif dic:
            super().__init__(dic)

if __name__ == "__main__":
    config = Config('./sample.yaml')
    print(config.data.dataset)