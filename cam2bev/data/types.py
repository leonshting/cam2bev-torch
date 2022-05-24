import typing as t


class WithIOInfo(t.Protocol):
    @property
    def num_in_channels(self) -> int:
        raise NotImplementedError

    @property
    def num_out_channels(self) -> int:
        raise NotImplementedError
