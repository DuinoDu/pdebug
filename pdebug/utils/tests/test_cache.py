from pdebug.utils.cache import FileCache


def test_cache_file(tmpdir):

    cache_file = f"{tmpdir}/cache.pkl"
    with FileCache(list, cache_file) as aa:
        aa.append(1)
        aa.append(2)

    with FileCache(list, cache_file) as bb:
        assert len(bb) == 2
        assert bb == aa
    print(bb)
