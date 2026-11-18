"""CPU tests for benchmark identity/protocol/reporting and worker control flow.
CUDA timing is mocked ONLY here; these numbers are never publication results.
"""
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('efficiency', Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_efficiency.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(project_root=str(self.root), dataset='isic18', seed=42,
            run_s='', run_a='', run_d='', data_root='', batch_size=0, gpus=1,
            warmup=1, steps=2, repeats=2, cached_batches=1)
        (self.root / 'pretrain.pt').write_bytes(b'pretrain')
        (self.root / 'teacher.pt').write_bytes(b'teacher')
        (self.root / 'images').mkdir()
        rows = [dict(split='train', sample_id=str(i), image='images/%d.png' % i,
                     mask='images/%d.png' % i, patient_id='') for i in range(4)]
        for i in range(4):
            (self.root / 'images' / ('%d.png' % i)).write_bytes(b'dummy')
        manifest = self.root / 'manifest.csv'
        with manifest.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        self.configs, self.runs = {}, {}
        for role in b.ROLES:
            cfg = dict(student_arch='shallow', seed=42, size=64, epochs=300, batch_size=2,
                       lr=.0001, min_lr=.000001, weight_decay=.01, augmentation=True,
                       rotation_degrees=15, mode='none' if role == 'S' else 'branch',
                       adapter='project' if role == 'S' else 'relation', beta=0 if role == 'S' else .1,
                       data_root=str(self.root), manifest=str(manifest), stage=3,
                       student_pretrained=str(self.root / 'pretrain.pt'),
                       teacher_checkpoint=str(self.root / 'teacher.pt'))
            if role != 'S':
                cfg.update(kd_stages=[2, 3], kd_stage_weights=[.5, .5])
            if role == 'D':
                cfg.update(scan_transfer_weight=.015, scan_transfer_variant='full')
            mode = 'baseline' if role == 'S' else 'branch_relation'
            run = self.root / 'runs' / 'isic18' / (b.RUN_NAMES[role][0] + '_' + mode + '_seed42')
            run.mkdir(parents=True)
            b.write_json(run / 'config.json', cfg)
            (run / 'best.pt').write_bytes(b'checkpoint')
            self.configs[role], self.runs[role] = cfg, run

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles_reject_endpoint_and_transfer_only(self):
        for role in b.ROLES:
            b.check_role(self.configs[role], role)
        for variant in ('endpoint', 'output'):
            cfg = dict(self.configs['D'], scan_transfer_variant=variant)
            with self.assertRaisesRegex(ValueError, 'FULL'):
                b.check_role(cfg, 'D')
        with self.assertRaises(ValueError):
            b.check_role(dict(self.configs['D'], scan_transfer_only=True), 'D')

    def test_plan_checks_protocol_and_never_writes_runs(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        plan = b.prepare_plan(self.args)
        self.assertEqual(plan['train_sample_ids'], ['0', '1'])
        self.assertEqual(plan['per_gpu_batch_size'], 2)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        cfg = dict(self.configs['D'], size=256)
        b.write_json(self.runs['D'] / 'config.json', cfg)
        with self.assertRaisesRegex(ValueError, 'Unmatched size'):
            b.prepare_plan(self.args)

    def test_manifest_content_and_ambiguity(self):
        alt = self.root / 'different.csv'
        alt.write_text((self.root / 'manifest.csv').read_text().replace('train,0,', 'test,0,'))
        b.write_json(self.runs['A'] / 'config.json', dict(self.configs['A'], manifest=str(alt)))
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            b.prepare_plan(self.args)
        second = self.root / 'runs/isic18/sab_A_relation_300_v1_branch_relation_seed42'
        second.mkdir()
        b.write_json(second / 'config.json', self.configs['A'])
        (second / 'best.pt').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'EFF_RUN_A'):
            b.resolve_run(self.root, 'isic18', 42, '', 'A')

    def test_batch_size_guard(self):
        self.args.gpus = 3
        with self.assertRaisesRegex(ValueError, 'divisible'):
            b.prepare_plan(self.args)

    def test_aggregate_uses_max_memory_and_sample_sd(self):
        plan = b.prepare_plan(self.args)
        output = self.root / 'report'
        output.mkdir()
        env = dict(torch='test', cuda_runtime='test', cudnn='test', devices=['test'], source_hashes={})
        for role in b.ROLES:
            for i in range(2):
                for phase in ('train', 'infer'):
                    b.write_json(output / ('%s_%s_%d.json' % (phase, role, i)),
                                 dict(max_rank_mean_ms=10 + 2 * i, mean_ms=1 + 2 * i,
                                      peak_allocated_bytes=(i + 1) * 2**30, student_parameters=1000000,
                                      environment=env))
        b.summarize(output, plan)
        summary = b.read_json(output / 'summary.json')
        row = summary['rows'][0]
        self.assertEqual(row['train_ms_step_mean'], 11)
        self.assertAlmostEqual(row['train_ms_step_std'], 2**.5)
        self.assertEqual(row['peak_train_GiB_per_GPU'], 2)
        self.assertEqual(row['student_parameters_M'], 1)
        self.assertEqual(summary['status'], 'complete')
        self.assertIn('D: Full method', (output / 'efficiency_table.md').read_text(encoding='utf-8'))

    def test_worker_forward_update_and_student_only_inference(self):
        import torch
        from torch import nn
        calls = []

        class TinySystem(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.student = nn.Conv2d(3, 1, 1)
                self.teacher = None if cfg['mode'] == 'none' else nn.Conv2d(3, 1, 1).requires_grad_(False)
                self.cfg = cfg

            def forward(self, image, target, alpha, beta):
                calls.append(('train', self.teacher is not None, beta))
                loss = (self.student(image) - target).square().mean()
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher = self.teacher(image)
                    loss = loss + beta * (self.student(image) - teacher).square().mean()
                return loss, loss.detach(), loss.detach(), []

            def predict(self, image):
                calls.append(('infer', self.teacher is not None, 0))
                return self.student(image)

        class TinyDataset:
            rows = [dict(sample_id=str(i)) for i in range(4)]

            def __getitem__(self, i):
                return torch.ones(3, 4, 4) * (i + 1), torch.ones(1, 4, 4), str(i)

        for role in b.ROLES:
            torch.save(dict(student=TinySystem(self.configs[role]).student.state_dict(),
                            config=self.configs[role]), self.runs[role] / 'best.pt')
        plan = b.prepare_plan(self.args)
        output = self.root / 'worker_report'
        output.mkdir()
        b.write_json(output / 'plan.json', plan)
        for role in b.ROLES:
            b.write_json(output / ('config_%s.json' % role), self.configs[role])
        runner = types.ModuleType('vmdmd.runner')
        runner.build_system = lambda cfg, initialize: (TinySystem(cfg), {})
        runner.coefficients = lambda cfg, epoch: (1., cfg['beta'])
        runner.dataset = lambda cfg, split: TinyDataset()
        runner.environment = lambda: {'devices': ['CPU mock, NOT measured GPU data']}
        runner.load_config = b.read_json
        runner.seed_all = lambda seed: torch.manual_seed(seed)
        cpu = torch.device('cpu')
        env = {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}
        original_checkpoints = [b.sha256(self.runs[r] / 'best.pt') for r in b.ROLES]
        with patch.dict(sys.modules, {'vmdmd': types.ModuleType('vmdmd'), 'vmdmd.runner': runner}), \
                patch.dict(b.os.environ, env), patch.object(torch, 'device', return_value=cpu), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'get_device_name', return_value='mock'), \
                patch.object(torch.cuda, 'set_device'), patch.object(torch.cuda, 'synchronize'), \
                patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                patch.object(torch.cuda, 'max_memory_allocated', return_value=100), \
                patch.object(torch.cuda, 'max_memory_reserved', return_value=200):
            for role in b.ROLES:
                for phase in ('train', 'infer'):
                    args = argparse.Namespace(plan=str(output / 'plan.json'), role=role, phase=phase,
                                              repeat=0, result=str(output / ('%s_%s.json' % (phase, role))))
                    b.run_worker(args)
                    result = b.read_json(args.result)
                    self.assertEqual(len(result['ranks'][0]['durations_ms']), 2)
                    self.assertEqual(result['student_parameters'], 4)
        self.assertTrue(any(kind == 'train' and teacher for kind, teacher, _ in calls))
        self.assertTrue(all(not teacher for kind, teacher, _ in calls if kind == 'infer'))
        self.assertEqual(original_checkpoints, [b.sha256(self.runs[r] / 'best.pt') for r in b.ROLES])


if __name__ == '__main__':
    unittest.main()
