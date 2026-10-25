"""Loopback-only Python network policy, installed before third-party imports.

This is a fail-closed application guard, not a kernel sandbox. Use strace alongside
it to audit native libraries and child processes as documented in the runbook.
"""
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import threading

_installed = False


def is_loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def install_local_network_guard(log_path):
    global _installed
    if _installed:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", buffering=1)
    lock = threading.Lock()

    def audit(event, args):
        if event in ("socket.connect", "socket.sendto"):
            sock, address = args[0], args[-1]
            if sock.family not in (socket.AF_INET, socket.AF_INET6):
                return
            host = address[0]
        elif event in ("socket.getaddrinfo", "socket.gethostbyname"):
            host = args[0]
            if host is None:
                return
            address = host
        else:
            return
        allowed = is_loopback(host)
        with lock:
            stream.write(json.dumps({"event": event, "address": address,
                                     "allowed": allowed, "pid": os.getpid()}) + "\n")
        if not allowed:
            raise PermissionError(f"Local-only network policy rejected {event}: {address!r}")

    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled"}.items():
        os.environ[key] = value
    os.environ.setdefault("YOLO_CONFIG_DIR", str(path.parent / "yolo-config"))
    sys.addaudithook(audit)
    _installed = True
