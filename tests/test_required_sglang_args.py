"""CPU regression tests for required MXFP8 launch arguments."""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('boot', Path(__file__).resolve().parents[1] / 'boot.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class RequiredArgsTests(unittest.TestCase):
    def resolve(self, args, backend='b12x'):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = boot.required_sglang_args(args, backend)
        return result, out.getvalue()

    def test_custom_args_keep_required_backend(self):
        args, warning = self.resolve(['--watchdog-timeout', '1800'])
        self.assertEqual(args[-2:], ['--fp8-gemm-backend', 'flashinfer_cutlass'])
        self.assertIn('WARNING', warning)

    def test_present_backend_is_not_duplicated(self):
        original = ['--fp8-gemm-backend', 'flashinfer_cutlass']
        args, warning = self.resolve(original)
        self.assertEqual(args, original)
        self.assertEqual(warning, '')

    def test_conflicting_backend_warns_and_is_replaced(self):
        for original in (['--fp8-gemm-backend', 'auto'],
                         ['--fp8-gemm-backend=auto'],
                         ['--fp8-gemm-backend', 'flashinfer_cutlass', '--fp8-gemm-backend=auto']):
            with self.subTest(original=original):
                args, warning = self.resolve(original)
                self.assertEqual(args, ['--fp8-gemm-backend', 'flashinfer_cutlass'])
                self.assertIn('auto', warning)

    def test_off_leaves_args_unchanged(self):
        original = ['--fp8-gemm-backend=auto']
        args, warning = self.resolve(original, 'off')
        self.assertEqual(args, original)
        self.assertEqual(warning, '')

    def test_boot_calls_resolver_after_extra_args(self):
        import inspect
        source = inspect.getsource(boot.serve)
        self.assertLess(source.index('args += extra.split()'), source.index('args = required_sglang_args('))

if __name__ == '__main__':
    unittest.main()
