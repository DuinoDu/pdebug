from pathlib import Path

from pdebug.utils.cache import FileCache, ResultCache


def test_cache_file(tmpdir):

    cache_file = f"{tmpdir}/cache.pkl"
    with FileCache(list, cache_file) as aa:
        aa.append(1)
        aa.append(2)

    with FileCache(list, cache_file) as bb:
        assert len(bb) == 2
        assert bb == aa
    print(bb)


def test_result_cache_fifo(tmp_path):
    cache = ResultCache("stage", chunk_size=2, tmp_root=Path(tmp_path))
    cache.append({"idx": 0})
    cache.append_many([{"idx": 1}, {"idx": 2}])
    cache.append({"idx": 3})
    cache.finalize()

    values = [entry["idx"] for entry in cache.iter_results()]
    assert values == [0, 1, 2, 3]
    assert cache.count == 4
    assert list(cache.tmpdir.glob("*.pkl"))

    cache.cleanup()
    assert not cache.tmpdir.exists()
