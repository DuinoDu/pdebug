"""Custom decorators"""
import functools
import inspect
import math
import multiprocessing as _mp
import os
import sys
import time
import traceback

__all__ = ["line_prof", "mp"]


def line_prof(decorated_=None, block=True, output="lp_output.txt.py"):
    """line_prof decorator

    Parameters
    ----------
    decorated_: None
        if @line_prof is used without (), this param is type of function
    block : bool, optional
        block and exit program in after line_profile
    output: str, optional
        line_profiler output log file
    """

    def _decorator(func):
        @functools.wraps(func)
        def __decorator(*args, **kwargs):
            import line_profiler

            profiler = line_profiler.LineProfiler(func)
            profiler.enable()
            res = func(*args, **kwargs)
            profiler.disable()
            fid = open(output, "w")
            profiler.print_stats(fid)
            fid.close()
            if block:
                print(
                    "exit when line_prof, disable by exit=False in line_prof"
                )
                print("saved to %s" % output)
                sys.exit()
            return res

        return __decorator

    if decorated_:
        return _decorator(decorated_)
    return _decorator


def _chunks(data_list, nums):
    num = int(math.ceil(len(data_list) / nums))
    for i in range(0, len(data_list), num):
        yield data_list[i : i + num]


def has_process_id(func):
    """Check if func has `process_id` args at first place."""
    # args = inspect.getargspec(func).args
    args = inspect.getfullargspec(func).args
    if "process_id" in args:
        assert (
            args.index("process_id") == 0
        ), "`process_id` can only be the first args"
        return True
    else:
        return False


def mp(decorated_=None, nums: int = None):
    """make your func run in multi-process

    Args:
        nums: process nums, default is 4

    """
    timeout = 10

    if decorated_ and nums:
        raise RuntimeError("Unexpected args.")

    def actual_decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            start = time.time()

            def compute_func(in_q, out_q):
                try:
                    result = func(*in_q.get())
                    out_q.put(result)
                except Exception as e:
                    print("Run func error: ", e)
                    raise RuntimeError

            if len(args) == 0:
                raise RuntimeError("Please provide input data list")

            if nums == 0:
                if has_process_id(func):
                    return func(0, *args)
                else:
                    return func(*args)

            input_index = -1
            for ind, arg in enumerate(args):
                if isinstance(arg, list):
                    input_index = ind
                    break
            if input_index == -1:
                raise TypeError("not found list in func args")

            # slice input list
            process_nums = os.cpu_count() if nums is None else nums
            data_lists = list(_chunks(args[input_index], process_nums))

            # create process
            in_qs = [_mp.Queue() for _ in range(process_nums)]
            out_qs = [_mp.Queue() for _ in range(process_nums)]
            process_pool = [
                _mp.Process(target=compute_func, args=(in_q, out_q))
                for in_q, out_q in zip(in_qs, out_qs)
            ]
            # run process
            try:
                for idx, data_list in enumerate(data_lists):
                    process_pool[idx].start()
                    input_args = []
                    for ind, arg in enumerate(args):
                        if ind == input_index:
                            input_args.append(data_list)
                        else:
                            input_args.append(arg)

                    # inject "process_id" into `input_args`
                    if has_process_id(func):
                        input_args.insert(0, idx)

                    input_args = tuple(input_args)
                    # may bugs
                    in_qs[idx].put(input_args)
            except Exception as e:
                traceback.print_exc()
                print("Run process error: ", e)
            # fetch result
            ret = []
            try:
                for out_q in out_qs:
                    ret_i = out_q.get(True)
                    if ret_i is None:
                        continue
                    elif isinstance(ret_i, (list, tuple)):
                        ret.extend(ret_i)
                    else:
                        ret.append(ret_i)
            except Exception as e:
                traceback.print_exc()
                print("Get process return value error: ", e)
            print(f"cost {time.time() - start} sec.")
            return ret

        return inner

    if decorated_:
        return actual_decorator(decorated_)
    return actual_decorator
