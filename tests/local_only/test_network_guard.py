import subprocess
import sys


def test_guard_allows_loopback_and_blocks_external_before_dns(tmp_path):
    code = '''
import socket, sys
from conceptgraph.utils.local_runtime import install_local_network_guard
install_local_network_guard(sys.argv[1])
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
client = socket.create_connection(server.getsockname(), timeout=1)
peer, _ = server.accept()
client.close(); peer.close(); server.close()
for operation in (
    lambda: socket.create_connection(("203.0.113.1", 443)),
    lambda: socket.getaddrinfo("api.openai.com", 443),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"test", ("203.0.113.1", 53)),
):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError("External network call was allowed")
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "network.jsonl")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
