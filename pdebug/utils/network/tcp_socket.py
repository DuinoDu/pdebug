import socket
import struct
import time

CHECK_SEND_CODE = "0123456789abcdefg"


class TcpSocket:
    def __init__(self, host: str, port: int, is_server=False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.host = host
        self.port = port
        self.is_server = is_server

    def start(self) -> bool:
        """Start server and waite for connection.Called by server."""
        assert self.is_server
        try:
            print(f"Tcp serving at {self.host}:{self.port}, waiting ...")
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            self.conn, addr = self.sock.accept()
            if self.conn:
                print(f"Connected by {addr}")
            return True
        except Exception as e:
            print(f"Start server failed: {e}")
            return False

    def connect(self) -> bool:
        """Connect to server. Called by client."""
        assert not self.is_server
        try:
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send(self, data: bytes, block=False) -> bool:
        """发送数据"""
        worker = self.conn if self.is_server else self.sock
        if not worker:
            if self.is_server:
                print("Server havs no connections")
            else:
                print("Not connected to a server")
            return False

        try:
            header = struct.pack("!I", len(data))
            worker.sendall(header + data)
            print(f"Sent {len(data)} bytes")

            if block:
                while True:
                    send_check_code = self.receive()
                    if (
                        send_check_code
                        and send_check_code.decode() == CHECK_SEND_CODE
                    ):
                        break

            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False

    def receive(self, buffer_size: int = 1024, block=False) -> bytes:
        """接收数据"""
        worker = self.conn if self.is_server else self.sock
        if not worker:
            if self.is_server:
                print("Server havs no connections")
            else:
                print("Not connected to a server")
            return b""

        try:
            header = worker.recv(4)
            if not header:
                return b""
            data_length = struct.unpack("!I", header)[0]
            received_data = b""
            while len(received_data) < data_length:
                packet = worker.recv(data_length - len(received_data))
                if not packet:
                    break
                received_data += packet

            # data = worker.recv(buffer_size)
            print(f"Received {len(received_data)} bytes")

            if block:
                time.sleep(0.1)
                self.send(CHECK_SEND_CODE.encode())

            return received_data
        except Exception as e:
            print(f"Receive failed: {e}")
            return b""

    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()
            print("Connection closed")


# 测试代码
if __name__ == "__main__":
    # 创建 TCP 客户端实例
    client = TCPClient()

    # 连接到服务器
    if client.connect("127.0.0.1", 65432):
        # 发送数据
        message = "Hello, Server!"
        if client.send(message.encode()):
            # 接收响应
            response = client.receive()
            print(f"Server response: {response.decode()}")

        # 关闭连接
        client.close()
    else:
        print("Failed to connect to the server")
