"""Check actual connect/send destination syscalls, including native/child code."""
import argparse
import ipaddress
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    destinations, rejected = set(), []
    for line in args.trace.read_text().splitlines():
        if not re.search(r"\b(connect|sendto|sendmsg|sendmmsg)\(", line):
            continue
        if "AF_INET" not in line:
            continue
        hosts = re.findall(r'inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)"', line)
        if not hosts:
            rejected.append(line)  # Unknown destination formatting must be reviewed.
        for ipv4, ipv6 in hosts:
            host = ipv4 or ipv6
            destinations.add(host)
            if not ipaddress.ip_address(host).is_loopback:
                rejected.append(line)
    report = {"status": "FAIL" if rejected else "PASS", "destinations": sorted(destinations),
              "external_or_unparsed": rejected, "scope": "connect/send syscalls in all strace-followed processes"}
    print(json.dumps(report, indent=2))
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
