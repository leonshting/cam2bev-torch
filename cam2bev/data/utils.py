import functools


def method_lru_cache(*cache_args, **cache_kwargs):
    def cache_decorator(func):
        @functools.wraps(func)
        def cache_factory(self, *args, **kwargs):
            instance_cache = functools.lru_cache(*cache_args, **cache_kwargs)(
                func,
            )
            instance_cache = instance_cache.__get__(self, self.__class__)
            setattr(self, func.__name__, instance_cache)
            return instance_cache(*args, **kwargs)

        return cache_factory

    return cache_decorator
