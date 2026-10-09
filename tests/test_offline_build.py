"""CPU-only build/pull regressions; all Docker and remote commands are stubs."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_launcher_build import ROOT, shell


class OfflineBuildTests(unittest.TestCase):
    def run_shell(self, code, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            return shell('ensure_ssh_keys() { :; }\n' + code, Path(tmp), settings)

    def test_skip_pull_uses_tag_without_digest(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          remote_on() { shift; if [[ "$1" == --timeout ]]; then shift 2; fi; bash -c "$(declare -f docker)
$*"; }
          cmd_pull
          cat "$LOG_DIR/commands"
        ''', SKIP_PULL='1', BASE_IMAGE='example/base:v1@sha256:123')
        self.assertNotIn('DOCKER pull', out)
        self.assertNotIn('@sha256:', '\n'.join(line for line in out.splitlines() if 'DOCKER' in line))
        self.assertEqual(out.count('DOCKER image inspect example/base:v1'), 3)

    def test_missing_local_base_fails_without_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*"; return 1; }
          if (cmd_pull) 2>&1; then echo PASSED; else echo FAILED; fi
        ''', SKIP_PULL='1')
        self.assertIn('FAILED', out)
        self.assertNotIn('DOCKER pull', out)
        self.assertIn('SKIP_PULL=1', out)

    def test_proxy_precheck_precedes_every_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          docker_proxy_precheck() { echo PROXY_CHECK; }
          remote_on() { shift; if [[ "$1" == --timeout ]]; then shift 2; fi; bash -c "$(declare -f docker docker_proxy_precheck)
$*"; }
          cmd_pull
          cat "$LOG_DIR/commands"
        ''')
        self.assertEqual(out.count('PROXY_CHECK'), 3)
        self.assertEqual(out.count('DOCKER pull'), 3)
        self.assertLess(out.index('PROXY_CHECK'), out.index('DOCKER pull'))

    def test_dead_proxy_stops_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          docker_proxy_precheck() { echo 'unreachable docker proxy'; return 1; }
          if (cmd_pull) 2>&1; then echo PASSED; else echo FAILED; fi
        ''')
        self.assertIn('FAILED', out)
        self.assertNotIn('DOCKER pull', out)

    def test_builds_pass_the_resolved_base_to_from(self):
        source = (ROOT / 'start.sh').read_text()
        build = source[source.index('cmd_build()'):source.index('cmd_share()')]
        self.assertIn('--build-arg', build)
        self.assertGreaterEqual(build.count('BASE_IMAGE='), 2)
        for name in ('Dockerfile', 'Dockerfile.canary', 'Dockerfile.canary-roce'):
            source = (ROOT / name).read_text()
            self.assertIn('ARG BASE_IMAGE=lmsysorg/sglang:dev-dsv41', source)
            self.assertIn('FROM ${BASE_IMAGE}', source)


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('proxy', ROOT / 'scripts/docker_proxy_check.py')
        cls.proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proxy)

    def test_no_proxy_is_inert(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({})
            connect.assert_not_called()

    def test_proxy_fields_timeout_and_credentials_redacted(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({'HttpProxy': 'http://secret:password@proxy.example:3128',
                                      'HttpsProxy': 'http://secret:password@proxy.example:3128'})
            connect.assert_called_once_with(('proxy.example', 3128), timeout=2)

    def test_unreachable_proxy_fails_clearly(self):
        with patch.object(self.proxy.socket, 'create_connection', side_effect=OSError('refused')):
            with self.assertRaisesRegex(RuntimeError, 'docker proxy proxy.example:3128 unreachable') as exc:
                self.proxy.check_proxies({'HTTPProxy': 'http://secret:password@proxy.example:3128'})
            self.assertNotIn('password', str(exc.exception))

if __name__ == '__main__':
    unittest.main()
