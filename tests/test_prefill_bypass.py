"""Mocked CPU scheduler with methods lifted from the real pinned source."""
import ast
import contextlib
import enum
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
STAGED = ROOT / 'runtime/sglang-canary/python/sglang/srt/managers'


class Result(enum.Enum):
    CONTINUE = 1
    OTHER = 2
    NO_TOKEN = 3


class Request:
    def __init__(self, name, fresh, prefix=0, output_budget=8):
        self.rid = name
        self.full_untruncated_fill_ids = [0] * (fresh + prefix)
        self.prefix_indices = [0] * prefix
        self.output_ids = []
        self.sampling_params = types.SimpleNamespace(ignore_eos=False, max_new_tokens=output_budget)
        self.host_hit_length = self.swa_host_hit_length = 0
        self.retracted_stain = False
        self.last_node = None
        self.beam_group = None
        self.session = None
        self.kv = types.SimpleNamespace(holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False)
        self.inflight_middle_chunks = 0
        self.set_extend_range(prefix, prefix + fresh)

    def set_extend_range(self, start, end):
        self.extend_range = types.SimpleNamespace(start=start, end=end, length=end-start)
        self.extend_input_len = end-start

    def init_next_round_input(self, *args):
        pass

    def needs_host_load_back(self):
        return False


class MemoryBudget:
    """Identical, explicit token accounting; no device/host memory query."""
    def __init__(self, capacity):
        self.remaining_total = self.remaining_current = capacity
        self.checks = []
        self.reservations = []
        self.partial_cap = None

    def check_prefill(self, **kwargs):
        self.checks.append(kwargs)
        fits = kwargs['total_tokens'] < self.remaining_total
        cap = kwargs['chunk_limit']
        if self.partial_cap is not None:
            cap = min(cap, self.partial_cap)
        return fits, cap if fits else None

    def has_capacity(self):
        return self.remaining_total > 0

    def reserve(self, fresh, max_new, **kwargs):
        self.reservations.append((fresh, max_new))
        self.remaining_total -= fresh + max_new + 1
        self.remaining_current = self.remaining_total

    def available_chunk_tokens(self, limit):
        return limit

    def fit_chunk(self, **kwargs):
        return kwargs['chunk_limit']


def lift(cls, filename, class_name, methods, namespace):
    """Compile original AST locations so inspect.getsource sees the actual pin."""
    tree = ast.parse(filename.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    for method in methods:
        node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == method)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(filename), 'exec'), namespace)
        setattr(cls, method, namespace[method])


def mocked_engine(capacity=100, partial_cap=None):
    memory = MemoryBudget(capacity)
    memory.partial_cap = partial_cap
    class Adder:
        def __init__(self, page_size, tree, allocator, running, ratio, rem_input, rem_chunk, *args, **kwargs):
            type(self).latest = self
            self.page_size = page_size
            self.tree_cache = tree
            self.memory_budget = memory
            self.running_batch = running
            self.rem_input_tokens = rem_input
            self.rem_chunk_tokens = rem_chunk
            self.dllm_config = None
            self.exact_chunk_fill = False
            self.can_run_list = []
            self.preempt_list = []
            self.new_chunked_req = None
            self.prefill_max_requests = None
            self.prefill_delayer_single_pass = None
            self.rem_mamba_slots = None
            self.log_hit_tokens = self.log_input_tokens = 0
        def _lock_node(self, *args):
            return contextlib.nullcontext()
        def _mamba_gap_budget_for_req(self, req):
            return 0
        def _req_inc_lock_ref(self, req):
            pass
        def _account_prefill_cache_admission(self, *args):
            pass
        def _check_prefill_tile_budget(self, *args):
            return None
    def admission(prefix, extend, max_new, chunked):
        return types.SimpleNamespace(prefix_len=prefix, extend_len=extend,
                                     max_new_tokens=max_new, is_chunked=chunked)
    namespace = dict(AddReqResult=Result, CLIP_MAX_NEW_TOKENS=4096, _PrefillAdmission=admission)
    lift(Adder, STAGED / 'schedule_policy.py', 'PrefillAdder',
         ['add_one_req', '_select_prefill_admission', '_check_prefill_budget', '_swa_new_tokens',
          '_commit_prefill_admission', '_update_prefill_budget', 'budget_state',
          'ceil_paged_tokens', 'add_chunked_req'], namespace)
    class Batch:
        def __init__(self, reqs=None, **kwargs):
            self.reqs = reqs or []
            self.batch_is_full = False
        @classmethod
        def init_new(cls, reqs, *args, **kwargs):
            return cls(reqs)
        def is_empty(self):
            return not self.reqs
        def prepare_for_extend(self):
            pass
    class Scheduler:
        pass
    namespace = dict(PrefillAdder=Adder, AddReqResult=Result, ScheduleBatch=Batch,
                     TEST_RETRACT=False, TEST_RETRACT_NO_PREFILL_BS=100,
                     DisaggregationMode=types.SimpleNamespace(PREFILL='prefill'),
                     get_schedule=lambda: types.SimpleNamespace(prefill_max_requests=None),
                     set_time_batch=lambda *a: None,
                     PrefillStats=types.SimpleNamespace(from_adder=lambda *a, **kw: None))
    lift(Scheduler, STAGED / 'scheduler.py', 'Scheduler', ['_get_new_batch_prefill_raw'], namespace)
    module = types.SimpleNamespace(Scheduler=Scheduler, PrefillAdder=Adder)
    scheduler = Scheduler()
    scheduler.grammar_manager = types.SimpleNamespace(has_waiting_grammars=lambda: False)
    for flag in ('enable_priority_preemption', 'enable_lora', 'enable_hicache_storage',
                 'enable_hierarchical_cache', 'enable_unified_cache_external_linker',
                 'enable_priority_scheduling', 'enable_overlap', 'is_mixed_chunk', 'is_hybrid_swa'):
        setattr(scheduler, flag, False)
    scheduler.disaggregation_mode = None
    scheduler.dllm_config = None
    scheduler.min_free_slots_delayer = None
    scheduler.dynamic_chunk_sizer = None
    scheduler.get_num_allocatable_reqs = lambda *a, **kw: 4
    scheduler.policy = types.SimpleNamespace(calc_priority=lambda *a, **kw: None)
    scheduler.processed_tokens_counter = 0
    scheduler.chunked_prefill_size = scheduler.max_prefill_tokens = 4
    scheduler.page_size = 1
    scheduler.max_prefill_bs = scheduler.max_running_requests = 4
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
        attn_backend=object(), prefill_aware_swa=False))
    scheduler.new_token_ratio_tracker = types.SimpleNamespace(current=1)
    scheduler.req_to_token_pool = types.SimpleNamespace()
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.tree_cache = types.SimpleNamespace(buffer_pipeline=None, storage_prefetch_retries=None, disable=True)
    scheduler.truncation_align_size = None
    scheduler.model_config = object()
    scheduler.spec_algorithm = object()
    scheduler.load_inquirer = types.SimpleNamespace(_get_num_pending_tokens=lambda **kw: 0)
    scheduler.chunked_req = Request('chunk', 20)
    scheduler.waiting_queue = [Request('long1', 16), Request('small', 2), Request('long2', 12)]
    return module, scheduler, Batch(), memory


@unittest.skipUnless((STAGED / 'scheduler.py').is_file(), 'stage pinned SGLang first')
class PrefillBypassTests(unittest.TestCase):
    def arm(self, module, value='4'):
        if os.environ.get('DSV41_TEST_STOCK_PREFILL'):
            return False
        adapter = importlib.import_module('prefill_bypass')
        with patch.dict(os.environ, {'DSV41_PREFILL_BYPASS': value}):
            return adapter.install(module)

    def test_small_admitted_during_chunk_and_queue_order_preserved(self):
        module, scheduler, running, memory = mocked_engine()
        self.arm(module)
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk', 'small'])
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ['long1', 'long2'])
        self.assertEqual(len(memory.checks), 1)
        self.assertEqual(memory.checks[0]['extend_input_len'], 2)
        self.assertEqual(memory.checks[0]['total_tokens'], 11)  # fresh + output + page
        self.assertEqual(memory.reservations[-1], (2, 8))
        self.assertEqual(module.PrefillAdder.latest.rem_chunk_tokens, -2)
        self.assertIsNone(module.PrefillAdder.latest.new_chunked_req)

    def test_rejected_small_candidate_does_not_block_smaller_one(self):
        module, scheduler, running, memory = mocked_engine(capacity=20)
        self.arm(module)
        scheduler.waiting_queue = [Request('too-expensive', 3, output_budget=100),
                                   Request('fits', 1), Request('long', 12)]
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk', 'fits'])
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ['too-expensive', 'long'])
        self.assertEqual(len(memory.checks), 2)
        self.assertEqual(module.PrefillAdder.latest.rem_chunk_tokens, -1)
        self.assertFalse(module.PrefillAdder.latest._dsv41_bypass_candidate)

    def test_memory_failure_keeps_all_waiting_requests(self):
        module, scheduler, running, memory = mocked_engine(capacity=8)
        self.arm(module)
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk'])
        self.assertEqual([r.rid for r in scheduler.waiting_queue], ['long1', 'small', 'long2'])
        self.assertTrue(memory.checks)
        self.assertEqual(memory.reservations, [(4, 0)])

    def test_partial_memory_fit_never_creates_second_chunked_request(self):
        module, scheduler, running, memory = mocked_engine(partial_cap=1)
        self.arm(module)
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk'])
        self.assertEqual(len(scheduler.waiting_queue), 3)

    def test_disabled_has_identical_symbols_and_behavior(self):
        module, scheduler, running, memory = mocked_engine()
        symbols = (module.Scheduler._get_new_batch_prefill_raw, module.PrefillAdder.add_one_req,
                   module.PrefillAdder._select_prefill_admission)
        self.assertFalse(self.arm(module, ''))
        self.assertEqual(symbols, (module.Scheduler._get_new_batch_prefill_raw,
                                  module.PrefillAdder.add_one_req, module.PrefillAdder._select_prefill_admission))
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk'])
        self.assertEqual(len(scheduler.waiting_queue), 3)

    def test_request_capacity_is_never_bypassed(self):
        module, scheduler, running, memory = mocked_engine()
        self.arm(module)
        scheduler.get_num_allocatable_reqs = lambda *a, **kw: 1
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk'])
        self.assertEqual(memory.checks, [])

    def test_no_chunk_has_stock_behavior(self):
        def run(enabled):
            module, scheduler, running, memory = mocked_engine()
            scheduler.chunked_req = None
            if enabled:
                self.arm(module)
            batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
            return [r.rid for r in batch.reqs], [r.rid for r in scheduler.waiting_queue], memory.reservations
        self.assertEqual(run(True), run(False))

    def test_cached_prefix_uses_fresh_tokens(self):
        module, scheduler, running, memory = mocked_engine()
        self.arm(module)
        scheduler.waiting_queue = [Request('cached-small', 2, prefix=100)]
        batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
        self.assertEqual([r.rid for r in batch.reqs], ['chunk', 'cached-small'])

    def test_four_ranks_choose_same_batch_with_different_unrelated_host_values(self):
        decisions = []
        for rank in range(4):
            module, scheduler, running, memory = mocked_engine()
            scheduler.unrelated_host_free_bytes = rank * 1000000
            self.arm(module)
            batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
            decisions.append(([r.rid for r in batch.reqs], [r.rid for r in scheduler.waiting_queue]))
        self.assertEqual(decisions, [(['chunk', 'small'], ['long1', 'long2'])] * 4)

    def test_drift_refuses_all_mutations(self):
        module, scheduler, running, memory = mocked_engine()
        original = module.PrefillAdder.add_one_req
        module.Scheduler._get_new_batch_prefill_raw = lambda *a: None
        self.assertFalse(self.arm(module))
        self.assertIs(module.PrefillAdder.add_one_req, original)

    def test_memory_check_symbol_drift_refuses_to_arm(self):
        module, scheduler, running, memory = mocked_engine()
        original = module.PrefillAdder.add_one_req
        module.PrefillAdder._check_prefill_budget = lambda *a, **kw: (True, None)
        self.assertFalse(self.arm(module))
        self.assertIs(module.PrefillAdder.add_one_req, original)

    def test_unsupported_preemption_mode_keeps_stock_behavior(self):
        def run(enabled):
            module, scheduler, running, memory = mocked_engine()
            scheduler.enable_priority_preemption = True
            if enabled:
                self.arm(module)
            batch, _ = scheduler._get_new_batch_prefill_raw(None, running)
            return [r.rid for r in batch.reqs], [r.rid for r in scheduler.waiting_queue], memory.reservations
        self.assertEqual(run(True), run(False))

    def test_sitecustomize_registers_scheduler(self):
        source = (ROOT / 'adapter/sitecustomize.py').read_text()
        self.assertEqual(source.count("'sglang.srt.managers.scheduler'"), 2)
        self.assertIn('install_prefill_bypass(module)', source)

    def test_invalid_limit_refuses_all_mutations(self):
        for value in ('0', '-1', 'bad'):
            module, scheduler, running, memory = mocked_engine()
            original = module.PrefillAdder.add_one_req
            self.assertFalse(self.arm(module, value))
            self.assertIs(module.PrefillAdder.add_one_req, original)

if __name__ == '__main__':
    unittest.main()
