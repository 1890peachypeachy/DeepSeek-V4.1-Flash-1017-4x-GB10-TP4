"""Fast TCP check of Docker daemon proxy endpoints; never print credentials."""
import json
import socket
import sys
from urllib.parse import urlsplit


def check_proxies(info):
    seen = set()
    for field in ('HttpProxy', 'HttpsProxy', 'HTTPProxy', 'HTTPSProxy'):
        value = info.get(field, '')
        if not value:
            continue
        parsed = urlsplit(value if '://' in value else 'http://' + value)
        try:
            endpoint = (parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))
        except ValueError:
            raise RuntimeError(f'docker {field}: invalid proxy port') from None
        if not endpoint[0]:
            raise RuntimeError(f'docker {field}: missing proxy host')
        if endpoint in seen:
            continue
        seen.add(endpoint)
        try:
            with socket.create_connection(endpoint, timeout=2):
                pass
        except OSError:
            raise RuntimeError(f'docker proxy {endpoint[0]}:{endpoint[1]} unreachable '
                               '(2s TCP timeout); fix daemon proxy or use SKIP_PULL=1') from None


if __name__ == '__main__':
    try:
        check_proxies(json.load(sys.stdin))
    except (RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
