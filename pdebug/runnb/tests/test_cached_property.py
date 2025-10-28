import shutil

from pdebug.runnb import cached_property

called = 0


def test_cached_property():
    class A:
        def __init__(self, mode):
            self.mode = mode
            self._a = None

        @cached_property
        def a(self):
            global called
            if self._a is None:
                self._a = 1 + 1
                print("computing")
                called = called + 1
            return self._a

    a = A("a")
    print(a.a)
    print(a.a)
    assert called == 1

    b = A("b")
    print(b.a)
    print(b.a)
    assert called == 2

    c = A("a")
    print(c.a)
    print(c.a)
    assert called == 2

    shutil.rmtree(".cached_property")
