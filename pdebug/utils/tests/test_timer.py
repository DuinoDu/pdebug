import time

from pdebug.utils.timer import Timer


def test_base():
    timer = Timer(enable=True)
    for i in range(3):
        timer.start()
        with timer.timeit("step1"):
            time.sleep(0.2)
        with timer.timeit("step2"):
            time.sleep(0.05)
        with timer.timeit("step3"):
            time.sleep(0.1)

        timer.start("step4")
        time.sleep(0.05)
        timer.stop("step4")

        time.sleep(0.1)
        timer.stop()
    timer.report()
