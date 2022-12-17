import weakref, itertools


class RememberPath:

    override_magic = {'__lt__', '__le__', '__gt__', '__ge__', '__ne__', '__eq__',

                      '__add__',
                      '__sub__',
                      '__mul__',
                      '__matmul__',
                      '__truediv__',
                      '__floordiv__',
                      '__mod__',
                      '__divmod__',
                      '__pow__',
                      '__lshift__',
                      '__rshift__',
                      '__and__',
                      '__xor__',
                      '__or__',

                      '__radd__',
                      '__rsub__',
                      '__rmul__',
                      '__rmatmul__',
                      '__rtruediv__',
                      '__rfloordiv__',
                      '__rmod__',
                      '__rdivmod__',
                      '__rpow__',
                      '__rlshift__',
                      '__rrshift__',
                      '__rand__',
                      '__rxor__',
                      '__ror__',

                      '__neg__',
                      '__pos__',
                      '__abs__',
                      '__invert__',
                      '__complex__',
                      '__int__',
                      '__float__',
                      '__index__',
                      '__round__',
                      '__trunc__',
                      '__floor__',
                      '__ceil__',

                      '__len__'}

    override_i_magic = {'__iadd__': lambda x, y: x + y,
                        '__isub__': lambda x, y: x - y,
                        '__imul__': lambda x, y: x * y,
                        '__imatmul__': lambda x, y: x @ y,
                        '__itruediv__': lambda x, y: x / y,
                        '__ifloordiv__': lambda x, y: x // y,
                        '__imod__': lambda x, y: x % y,
                        '__ipow__': lambda x, y: x ** y,
                        '__ilshift__': lambda x, y: x << y,
                        '__irshift__': lambda x, y: x >> y,
                        '__iand__': lambda x, y: x & y,
                        '__ixor__': lambda x, y: x ^ y,
                        '__ior__': lambda x, y: x | y, }

    exclude = {}


    def __init__(self, x=None, is_fn=False,
                 # mut_methods: when attributes within are called, self may change
                 mut_methods: set[str] = (),
                 args: tuple = (), kwargs: dict = None, root=None): # root: typing: Self for python ver. >= 3.11
        assert not isinstance(x, RememberPath)

        self._i_rpfwm_ = x
        self._is_fn_rpfwm_ = is_fn
        # values depending on self
        self._children_rpfwm_ = weakref.WeakSet[RememberPath]()

        self._args_rpfwm_ = [RememberPath._to_fe_rpfwm_(a) for a in args]
        self._kwargs_rpfwm_ = {k: RememberPath._to_fe_rpfwm_(v) for k, v in kwargs.items()} if kwargs is not None else {}

        self._dirty_rpfwm_ = is_fn
        self._value_rpfwm_ = None if is_fn else x

        self._mut_methods_rpfwm_ = mut_methods
        self._root_rpfwm_: RememberPath = self if root is None else root


    @staticmethod
    def _to_fe_rpfwm_(x):
        if isinstance(x, RememberPath):
            return x
        else:
            return RememberPath(x)


    def mark_dirty(self):
        self._dirty_rpfwm_ = self._is_fn_rpfwm_

        for d in self._children_rpfwm_:
            if d is not None and not d._dirty_rpfwm_:
                d.mark_dirty()


    def __getitem__(self, item):

        root = self if isinstance(item, RememberPath) and item._root_rpfwm_._is_fn_rpfwm_ else self._root_rpfwm_

        ret = RememberPath(lambda it: self.i[it], is_fn=True, root=root, args=(item,))
        object.__getattribute__(self, '_children_rpfwm_').add(ret)

        return ret


    def __setitem__(self, key, value):

        assert not (self._root_rpfwm_._is_fn_rpfwm_ or isinstance(value, RememberPath))
        ret = self.i.__setitem__(key, value)
        self._root_rpfwm_.mark_dirty()
        return ret


    def __setattr__(self, key: str, value):

        if not RememberPath.exclude or key in RememberPath.exclude:
            return object.__setattr__(self, key, value)

        assert not (self._root_rpfwm_._is_fn_rpfwm_ or isinstance(value, RememberPath))
        ret = self.i.__setattr__(key, value)
        self._root_rpfwm_.mark_dirty()
        return ret


    def _attr_i_call_rpfwm_(self, name: str, other):
        assert not (self._root_rpfwm_._is_fn_rpfwm_ or isinstance(other, RememberPath))

        selfie = self.i
        if hasattr(selfie, name):
            ret = getattr(selfie, name)(other)
            self.mark_dirty()
            return ret
        else:
            self.i = RememberPath.override_i_magic[name](selfie, other)


    @property
    def i(self):
        if self._dirty_rpfwm_:
            assert self._is_fn_rpfwm_
            self._value_rpfwm_ = self._i_rpfwm_(*(a.i for a in self._args_rpfwm_),
                                                **{k: v.i for k, v in self._kwargs_rpfwm_.items()})

        self._dirty_rpfwm_ = False
        return self._value_rpfwm_


    @i.setter
    def i(self, value):
        assert not isinstance(value, RememberPath)

        self.mark_dirty()

        self._i_rpfwm_ = value

        self._is_fn_rpfwm_ = False

        self._args_rpfwm_ = []
        self._kwargs_rpfwm_ = {}

        self._dirty_rpfwm_ = False
        self._value_rpfwm_ = value


    def __getattr__(self, item: str):

        if item in RememberPath.exclude:
            return object.__getattribute__(self, item)

        if item in object.__getattribute__(self, '_mut_methods_rpfwm_'):
            assert not object.__getattribute__(self._root_rpfwm_, '_is_fn_rpfwm_')

            def g(*args, **kwargs):
                r = getattr(self.i, item)(*args, **kwargs)
                self._root_rpfwm_.mark_dirty()
                return r

            return g

        ret = RememberPath(lambda: getattr(self.i, item), is_fn=True, root=self)
        object.__getattribute__(self, '_children_rpfwm_').add(ret)

        return ret


    def __call__(self, *args, **kwargs):

        ret = RememberPath(lambda *a, **ka: self.i(*a, **ka), is_fn=True, args=args, kwargs=kwargs)

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, RememberPath):
                arg._children_rpfwm_.add(ret)

        self._children_rpfwm_.add(ret)

        return ret


RememberPath.exclude = RememberPath.__dict__.keys() | RememberPath().__dict__.keys()


def init_class():

    def c(name: str):
        return lambda self, other: self._attr_i_call_rpfwm_(name, other)

    for name in RememberPath.override_i_magic:
        setattr(RememberPath, name, c(name))

    def d(name: str):
        return lambda self, *args, **kwargs: RememberPath.__getattr__(self, name)(*args, **kwargs)

    for name in RememberPath.override_magic:
        setattr(RememberPath, name, d(name))


init_class()
del init_class


if __name__ == '__main__':

    p = RememberPath

    a, b = p(0), p(0)
    result = 1 + a * (a + b + b * b + 0)
    assert result.i == 1

    a.i = 1  # b * b is cached
    assert result.i == 1 + 1 * (1 + 0 + 0 * 0 + 0)

    b.i = 2
    assert result.i == 1 + 1 * (1 + 2 + 2 * 2 + 0)

    a.i, b.i = 1, 3
    a += 1
    assert result.i == 1 + 2 * (2 + 3 + 3 * 3 + 0)

    import numpy as np
    import pandas as pd

    npe = p(np)
    a = p(np.zeros(5))
    b = p(np.zeros(5))
    c = np.zeros(5)
    r = npe.outer(a, a + b) + c
    print(r.i)
    a.i = np.ones(5)
    print(r.i)

    df = p(pd.DataFrame())
    dfp = df + 100
    print(df.i)
    df.index = [0, 1]
    df['r'] = [2, 3]
    print(df.i)
    df.iloc[1] = 9
    print(dfp.i)
    df += 1000
    df.iloc[0] -= 5000
    print(dfp.i)

    '''
    [[0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]]
    [[1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1.]
      [1. 1. 1. 1. 1.]]
    Empty DataFrame
    Columns: []
    Index: []
        r
    0  2
    1  3
          r
    0  102
    1  109
          r
    0 -3898
    1  1109
    '''
