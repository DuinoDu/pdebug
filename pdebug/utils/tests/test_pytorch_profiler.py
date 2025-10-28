from pdebug.utils.pytorch_profiler import pretty_print_list


def test_pretty_print_list():
    a = pretty_print_list()
    a.append(1)
    a.append(2)
    a.append(3)
    print(a)
