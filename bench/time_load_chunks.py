import contextlib
import io
import itertools
import os
import time
import zlib

import numpy as np
import rapidgzip

from concatenated_zlib import zlib_concat_decode
from concatenated_zlib import zlibng_concat_decode
from concatenated_zlib import zlib_multi_decompress
from concatenated_zlib._deflate import libdeflate_zlib_decode

def test():

    c0 = open("./chunk_0000.dat", "rb").read()
    c1 = open("./chunk_0001.dat", "rb").read()

    d0 = zlib.decompress(c0)
    d1 = zlib.decompress(c1)

    assert d0 == zlib_concat_decode(c0)
    assert d1 == zlib_concat_decode(c1)
    assert (d0 + d1) == zlib_concat_decode(c0 + c1)

    assert d0 == zlibng_concat_decode(c0)
    assert d1 == zlibng_concat_decode(c1)
    assert (d0 + d1) == zlibng_concat_decode(c0 + c1)

    assert d0 == zlib_multi_decompress([c0])
    assert d1 == zlib_multi_decompress([c1])
    assert (d0 + d1) == zlib_multi_decompress([c0, c1])

    print(d0 + d1 == zlib_concat_decode(c0 + c1))
    print(d0 + d1 == zlibng_concat_decode(c0 + c1))
    print(d0 + d1 == zlib_multi_decompress([c0, c1]))

@contextlib.contextmanager
def timeit(label, div=1):
    t0 = time.monotonic()
    try:
        yield
    finally:
        print(label, "took", round((time.monotonic() - t0) / div, 3), "seconds")


def bench():

    c0 = open("./chunk_0000.dat", "rb").read()
    c1 = open("./chunk_0001.dat", "rb").read()
    test_bytes_list = lambda: itertools.islice(itertools.cycle([c0, c1]), 40000)
    test_bytes_concatenated = b"".join(test_bytes_list())
    print(f"Concatenated size: {len(test_bytes_concatenated) / 1e6} MB")

    outputFilePath = "many-concatenated.zlib"
    with open(outputFilePath, 'wb') as file:
        file.write(test_bytes_concatenated)

    with timeit("overhead"):
        x = test_bytes_concatenated

    with timeit("stdlib.zlib"):
        _data0 = b"".join(zlib.decompress(c) for c in test_bytes_list())

    with timeit("stdlib.zlib without joining"):
        for c in test_bytes_list():
            zlib.decompress(c)

    for parallelization in [1, 2, 4, 8, os.cpu_count()]:
        chunkSize = 1024 * 1024
        with timeit(f"rapidgzip_decompress from BytesIO read and discard {chunkSize / 1024 / 1024} MiB chunks "
                    f"parallelization={parallelization}"):
            with rapidgzip.RapidgzipFile(io.BytesIO(test_bytes_concatenated), parallelization=parallelization) as file:
                while result := file.read(chunkSize):
                    pass

    for parallelization in [1, 2, 4, 8, os.cpu_count()]:
        with timeit(f"rapidgzip_decompress from BytesIO parallelization={parallelization}"):
            with rapidgzip.RapidgzipFile(io.BytesIO(test_bytes_concatenated), parallelization=parallelization) as file:
                _data5 = file.read()
        assert _data0 == _data5
        del _data5

    for parallelization in [1, 2, 4, 8, os.cpu_count()]:
        with timeit(f"rapidgzip_decompress from file parallelization={parallelization}"):
            with rapidgzip.RapidgzipFile(outputFilePath, parallelization=parallelization) as file:
                _data5 = file.read()
        assert _data0 == _data5
        del _data5

    with timeit("zlib_concat_decode"):
        _data1 = zlib_concat_decode(test_bytes_concatenated)
    assert _data0 == _data1
    del _data1

    with timeit("zlibng_concat_decode"):
        _data2 = zlibng_concat_decode(test_bytes_concatenated)
    assert _data0 == _data2
    del _data2

    with timeit("zlib_multi_decompress"):
        _data3 = zlib_multi_decompress(test_bytes_list())
    assert _data0 == _data3
    del _data3

    with timeit("libdeflate_zlib_decompress"):
        out = np.empty((1024*12 + 24,), dtype=np.uint8)
        _data4 = b"".join(libdeflate_zlib_decode(c, out) for c in test_bytes_list())
    assert _data0 == _data4
    del _data4

    # with timeit("onecall"):
    #    z.zlib_decode(b"".join(itertools.islice(itertools.cycle([c0, c1]), 256000)))


if __name__ == "__main__":
    test()
    bench()


# Concatenated size: 196.92 MB
# overhead took 0.0 seconds
# stdlib.zlib took 1.97 seconds
# stdlib.zlib without joining took 1.422 seconds
# rapidgzip_decompress from BytesIO read and discard 1.0 MiB chunks parallelization=1 took 1.406 seconds
# rapidgzip_decompress from BytesIO read and discard 1.0 MiB chunks parallelization=2 took 0.775 seconds
# rapidgzip_decompress from BytesIO read and discard 1.0 MiB chunks parallelization=4 took 0.439 seconds
# rapidgzip_decompress from BytesIO read and discard 1.0 MiB chunks parallelization=8 took 0.365 seconds
# rapidgzip_decompress from BytesIO read and discard 1.0 MiB chunks parallelization=16 took 0.361 seconds
# rapidgzip_decompress from BytesIO parallelization=1 took 2.098 seconds
# rapidgzip_decompress from BytesIO parallelization=2 took 1.603 seconds
# rapidgzip_decompress from BytesIO parallelization=4 took 1.048 seconds
# rapidgzip_decompress from BytesIO parallelization=8 took 1.166 seconds
# rapidgzip_decompress from BytesIO parallelization=16 took 1.312 seconds
# rapidgzip_decompress from file parallelization=1 took 2.088 seconds
# rapidgzip_decompress from file parallelization=2 took 1.669 seconds
# rapidgzip_decompress from file parallelization=4 took 0.988 seconds
# rapidgzip_decompress from file parallelization=8 took 1.133 seconds
# rapidgzip_decompress from file parallelization=16 took 1.27 seconds
# zlib_concat_decode took 1.954 seconds
# zlibng_concat_decode took 1.635 seconds
# zlib_multi_decompress took 2.018 seconds
# libdeflate_zlib_decompress took 1.374 seconds
