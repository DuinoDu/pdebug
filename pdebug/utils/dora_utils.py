from pdebug.utils.env import DORA_INSTALLED

if DORA_INSTALLED:
    from dora import Node


def operator_main(operator_class) -> None:
    """
    Run dora operator in dynamic(debug) mode.
    """
    assert DORA_INSTALLED, "dora is required."

    op = operator_class()
    op_name = getattr(operator_class, "name", "my_operator")
    node = Node(op_name)
    for event in node:
        e_type = event["type"]
        if e_type == "INPUT":
            op.on_event(event, node.send_output)
        elif e_type == "STOP":
            print(f"[{op_name}] received stop")
        elif e_type == "ERROR":
            print(f"[{op_name}] error: ", event["error"])
        elif e_type == "INPUT_CLOSED":
            print(f"[{op_name}] received input_closed")
        else:
            print(f"[{op_name}] received unexpected event: ", e_type)
