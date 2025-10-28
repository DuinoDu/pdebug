from pdebug.utils.network import TcpSocket

import pytest

"""
>> pytest -s pdebug/utils/tests/test_network.py::test_tcpsocket_server
>> pytest -s pdebug/utils/tests/test_network.py::test_tcpsocket_client
"""

host = "127.0.0.1"
ip = 12346


@pytest.mark.skip("tested")
def test_tcpsocket_server():
    server = TcpSocket(host, ip, True)
    server.start()

    while True:
        data = server.receive()
        if not data:
            break
        print(f"Received: {data.decode()}")
        server.send(b"Server received your message")
    server.close()


@pytest.mark.skip("tested")
def test_tcpsocket_client():
    client = TcpSocket(host, ip, False)
    client.connect()

    message = "Hello, Server!"
    if client.send(message.encode()):
        response = client.receive()
        print(f"Server response: {response.decode()}")
    client.close()


@pytest.mark.skip("tested")
def test_tcpsocket_server_block():
    server = TcpSocket(host, ip, True)
    server.start()

    while True:
        data = server.receive(block=True)
        if not data:
            break
        print(f"Received: {data.decode()}")
        server.send(b"Server received your message")
    server.close()


@pytest.mark.skip("tested")
def test_tcpsocket_client_block():
    client = TcpSocket(host, ip, False)
    client.connect()

    message = "Hello, Server!"
    if client.send(message.encode(), block=True):
        response = client.receive()
        print(f"Server response: {response.decode()}")
    client.close()
