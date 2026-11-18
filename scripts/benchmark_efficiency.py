"""S/A/paper-D efficiency benchmark. Python 3.8; never writes training runs.

Coordinator uses only stdlib. Each measurement runs in a fresh process.
Training starts from the same pretrained student (not a resumed optimizer).
Inference loads each run's best.pt and constructs ONLY the student.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROLES = ('S', 'A', 'D')
RUN_NAMES = {
    'S': ('sab_S_baseline_300_v1',),
    'A': ('stage34_relation_300_control_v1', 'sab_A_relation_300_v1'),
    'D': ('scan_transfer_stage3_300_v1', 'sab_B_transfer_300_v1'),
}
PATH_KEYS = ('data_root', 'manifest', 'student_pretrained', 'teacher_checkpoint')
COMMON_KEYS = ('student_arch', 'size', 'seed', 'epochs', 'batch_size', 'lr',
               'min_lr', 'weight_decay', 'augmentation', 'rotation_degrees')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_run(root, dataset, seed, explicit, role):
    if explicit:
        p = Path(explicit).expanduser()
        found = [p if p.is_absolute() else root / 'runs' / dataset / p]
    else:
        mode = 'baseline' if role == 'S' else 'branch_relation'
        found = [root / 'runs' / dataset / ('%s_%s_seed%d' % (n, mode, seed))
                 for n in RUN_NAMES[role]]
        found = [p for p in found if (p / 'config.json').is_file() and (p / 'best.pt').is_file()]
    if len(found) != 1:
        raise ValueError('Set EFF_RUN_%s to the exact run folder. Found: %s' % (role, found))
    run = found[0].resolve()
    for name in ('config.json', 'best.pt'):
        if not (run / name).is_file():
            raise FileNotFoundError(str(run / name))
    return run


def check_role(cfg, role):
    if cfg.get('student_arch') != 'shallow':
        raise ValueError(role + ': expected the shallow VM-UNet student')
    for key in ('task_gradient', 'prototype_weight', 'cross_scale_weight', 'collective_mode',
                'collective_aux_weight', 'relation_curriculum', 'relation_reliability',
                'kd_direction_weights', 'relation_component_weights', 'scan_transfer_only'):
        if cfg.get(key):
            raise ValueError(role + ': unexpected additional supervision: ' + key)
    weight = cfg.get('scan_transfer_weight', 0.)
    if role == 'S':
        if cfg.get('mode') != 'none' or cfg.get('beta') != 0 or weight != 0:
            raise ValueError('S must have no distillation')
    else:
        if (cfg.get('mode') != 'branch' or cfg.get('adapter') != 'relation'
                or cfg.get('kd_stages') != [2, 3] or cfg.get('kd_stage_weights') != [.5, .5]
                or cfg.get('beta') != .1 or cfg.get('relation_radius', 1) != 1):
            raise ValueError(role + ': expected stage34 branch relation, beta=0.1')
        if role == 'A' and weight != 0:
            raise ValueError('A must not include internal transfer supervision')
        if role == 'D' and (not math.isfinite(weight) or weight <= 0
                            or cfg.get('scan_transfer_variant', 'full') != 'full'):
            raise ValueError('Paper D must be FULL transfer (old train_B), not endpoint/output')


def manifest_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'split', 'sample_id', 'image', 'mask', 'patient_id'}
    if not rows or not required <= set(rows[0]):
        raise ValueError('Invalid manifest: ' + str(path))
    # Compare content, not file names; each run can have its own identical manifest.
    return sorted([{k: r[k] for k in sorted(required)} for r in rows],
                  key=lambda r: (r['split'], r['sample_id']))


def prepare_plan(args):
    root = Path(args.project_root).resolve()
    records = {}
    for role in ROLES:
        run = resolve_run(root, args.dataset, args.seed, getattr(args, 'run_' + role.lower()), role)
        cfg = read_json(run / 'config.json')
        check_role(cfg, role)
        if cfg['seed'] != args.seed:
            raise ValueError(role + ': config seed differs from requested seed')
        for key in PATH_KEYS:
            if cfg.get(key):
                p = Path(os.path.expandvars(cfg[key])).expanduser()
                cfg[key] = str((p if p.is_absolute() else root / p).resolve())
        if args.data_root:
            cfg['data_root'] = str(Path(args.data_root).resolve())
        # Archived runs often retain a snapshot even if the old data/*.csv was moved.
        if not Path(cfg['manifest']).is_file() and (run / 'manifest.csv').is_file():
            cfg['manifest'] = str(run / 'manifest.csv')
        required = ['manifest', 'student_pretrained'] + ([] if role == 'S' else ['teacher_checkpoint'])
        for key in required:
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(role + ': ' + key + '=' + cfg[key])
        if not Path(cfg['data_root']).is_dir():
            raise FileNotFoundError(cfg['data_root'])
        records[role] = dict(run=str(run), config=cfg, config_sha256=sha256(run / 'config.json'),
                             checkpoint=str(run / 'best.pt'), checkpoint_sha256=sha256(run / 'best.pt'))
    base = records['S']['config']
    rows = manifest_rows(base['manifest'])
    for role in ('A', 'D'):
        cfg = records[role]['config']
        for key in COMMON_KEYS + ('data_root',):
            if cfg.get(key) != base.get(key):
                raise ValueError('Unmatched %s: S=%r, %s=%r' % (key, base.get(key), role, cfg.get(key)))
        if manifest_rows(cfg['manifest']) != rows:
            raise ValueError(role + ': split/image/mask manifest differs from S')
    pretrained = {sha256(v['config']['student_pretrained']) for v in records.values()}
    if len(pretrained) != 1:
        raise ValueError('Student pretrained weights differ')
    teachers = [sha256(records[r]['config']['teacher_checkpoint']) for r in ('A', 'D')]
    if teachers[0] != teachers[1]:
        raise ValueError('A and D teacher checkpoints differ')
    batch = args.batch_size or base['batch_size']
    if batch % args.gpus or batch < args.gpus:
        raise ValueError('Global batch size must be divisible by the training GPU count')
    train_rows = [r for r in rows if r['split'] == 'train']
    if len(train_rows) < batch * args.cached_batches:
        raise ValueError('Not enough train images for the requested CPU batch bank')
    for r in train_rows[:batch * args.cached_batches]:
        for key in ('image', 'mask'):
            if not (Path(base['data_root']) / r[key]).is_file():
                raise FileNotFoundError(str(Path(base['data_root']) / r[key]))
    return dict(project_root=str(root), dataset=args.dataset, seed=args.seed, records=records,
                training_gpus=args.gpus, global_batch_size=batch, per_gpu_batch_size=batch // args.gpus,
                batch_size_override=batch != base['batch_size'], size=base['size'], precision='float32',
                warmup=args.warmup, steps=args.steps, repeats=args.repeats, cached_batches=args.cached_batches,
                inference_batch_size=1, inference_gpus=1, benchmark_epoch=1,
                student_pretrained_sha256=next(iter(pretrained)), teacher_checkpoint_sha256=teachers[0],
                train_sample_ids=[r['sample_id'] for r in train_rows[:batch * args.cached_batches]],
                training_initialization='same student pretrain, fresh AdamW; full configured KD weights; no schedule',
                training_scope='normalization, teacher if used, student, losses, backward, AdamW and DDP; excludes IO/H2D/logging/validation/checkpointing',
                inference_scope='student forward plus ImageNet normalization; excludes IO/H2D/resize/threshold; no teacher or KD',
                timing='per-step synchronized wall time; training uses maximum rank mean',
                memory='max_memory_allocated after warmup; maximum across GPUs, then repeats; GiB=2**30',
                batch_bank='deterministic augmented train images cached on CPU, cycled identically; no synthetic training masks')


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.)


def summarize(output, plan):
    rows = []
    for role in ROLES:
        train = [read_json(output / ('train_%s_%d.json' % (role, i))) for i in range(plan['repeats'])]
        infer = [read_json(output / ('infer_%s_%d.json' % (role, i))) for i in range(plan['repeats'])]
        ts = stats([v['max_rank_mean_ms'] for v in train])
        ins = stats([v['mean_ms'] for v in infer])
        params = {v['student_parameters'] for v in train + infer}
        if len(params) != 1:
            raise ValueError('Student parameter count differs between train and inference')
        rows.append(dict(method=role, train_ms_step_mean=ts['mean'], train_ms_step_std=ts['std'],
                         peak_train_GiB_per_GPU=max(v['peak_allocated_bytes'] for v in train) / 2**30,
                         student_parameters=next(iter(params)), student_parameters_M=next(iter(params)) / 1e6,
                         inference_ms_image_mean=ins['mean'], inference_ms_image_std=ins['std']))
    if len({r['student_parameters'] for r in rows}) != 1:
        raise ValueError('S/A/D architectures have different parameter counts')
    envs = [read_json(output / ('%s_%s_%d.json' % (phase, role, i)))['environment']
            for phase in ('train', 'infer') for role in ROLES for i in range(plan['repeats'])]
    for key in ('torch', 'cuda_runtime', 'cudnn', 'devices', 'source_hashes'):
        if any(env[key] != envs[0][key] for env in envs[1:]):
            raise ValueError('Environment changed between measurements: ' + key)
    with (output / 'efficiency_table.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = dict(S='Student without KD', A='Branch Local Relation KD', D='Full method')
    lines = ['| Method | Train (ms/step) | Peak allocated (GiB/GPU) | Student params (M) | Inference (ms/image) |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {name} | {train_ms_step_mean:.3f} ± {train_ms_step_std:.3f} | {peak_train_GiB_per_GPU:.3f} | {student_parameters_M:.3f} | {inference_ms_image_mean:.3f} ± {inference_ms_image_std:.3f} |'.format(name=r['method'] + ': ' + names[r['method']], **r))
    lines += ['', 'Mean ± sample SD over %d fresh-process repetitions (not accuracy seeds).' % plan['repeats'],
              'FP32, input %dx%d; train %d GPU(s), global batch %d; inference 1 GPU, batch 1.' %
              (plan['size'], plan['size'], plan['training_gpus'], plan['global_batch_size']),
              'GPU: ' + ', '.join(envs[0]['devices']), plan['training_scope'], plan['inference_scope'],
              'Training is a short compute benchmark, not full-epoch or end-to-end training wall time.',
              'Peak memory includes model/teacher/optimizer/activations and is maximum across ranks and repetitions.']
    (output / 'efficiency_table.md').write_text('\n\n'.join(['\n'.join(lines[:5]), *lines[6:]]) + '\n', encoding='utf-8')
    a, d = rows[1], rows[2]
    write_json(output / 'summary.json', dict(status='complete', rows=rows, protocol=plan,
               full_vs_relation_train_overhead_percent=100 * (d['train_ms_step_mean'] / a['train_ms_step_mean'] - 1),
               full_vs_relation_memory_overhead_percent=100 * (d['peak_train_GiB_per_GPU'] / a['peak_train_GiB_per_GPU'] - 1)))


def run_worker(args):
    # Imports are delayed so planning and report tests run without CUDA dependencies.
    plan = read_json(args.plan)
    sys.path.insert(0, plan['project_root'])
    os.environ['VMDMD_SCAN_BACKEND'] = 'cuda'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from vmdmd.runner import build_system, coefficients, dataset, environment, load_config, seed_all

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA GPU and the CUDA selective-scan backend')
    if torch.cuda.device_count() < plan['training_gpus']:
        raise ValueError('Fewer visible GPUs than requested; match sbatch --gpus and EFF_GPUS')
    if len({torch.cuda.get_device_name(i) for i in range(plan['training_gpus'])}) != 1:
        raise ValueError('Use the same GPU model for every training rank')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.phase == 'infer' and world != 1:
        raise ValueError('Inference must run in a single process')
    if args.phase == 'train' and world != plan['training_gpus']:
        raise ValueError('torchrun world size differs from plan')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    cfg = load_config(Path(args.plan).parent / ('config_%s.json' % args.role))
    seed_all(plan['seed'])
    record = plan['records'][args.role]
    if sha256(Path(record['run']) / 'config.json') != record['config_sha256']:
        raise ValueError('Saved training config changed after benchmark planning')
    if args.phase == 'train':
        system, load_report = build_system(cfg, initialize=True)
        system.to(device).train()
        optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],
                                      lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        model = DDP(system, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else system
        train = dataset(cfg, 'train')
        train.epoch = plan['benchmark_epoch']
        index = {r['sample_id']: i for i, r in enumerate(train.rows)}
        bank = []
        local_batch = plan['per_gpu_batch_size']
        for b in range(plan['cached_batches']):
            start = b * plan['global_batch_size'] + rank * local_batch
            ids = plan['train_sample_ids'][start:start + local_batch]
            items = [train[index[sid]] for sid in ids]
            bank.append((torch.stack([v[0] for v in items]), torch.stack([v[1] for v in items])))
        seed_all(plan['seed'] + rank)
        alpha, beta = coefficients(cfg, plan['benchmark_epoch'])

        def step(i, measure):
            image, target = [v.to(device) for v in bank[i % len(bank)]]
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(image, target, alpha, beta)
            loss = result[0]
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - begin) * 1000
            return elapsed
    else:
        # Disable all auxiliary paths at construction; no teacher is instantiated.
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg['mode'] = 'none'
        system, load_report = build_system(eval_cfg, initialize=False)
        checkpoint = torch.load(record['checkpoint'], map_location='cpu')
        if sha256(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after benchmark planning')
        if 'student' not in checkpoint or 'config' not in checkpoint:
            raise ValueError('Inference requires a best.pt with student and config')
        original_cfg = read_json(Path(record['run']) / 'config.json')
        if {k: v for k, v in checkpoint['config'].items() if k not in PATH_KEYS} != {
                k: v for k, v in original_cfg.items() if k not in PATH_KEYS}:
            raise ValueError('best.pt config does not match saved config.json')
        system.student.load_state_dict(checkpoint['student'], strict=True)
        del checkpoint
        system.to(device).eval()
        if system.teacher is not None:
            raise AssertionError('Inference unexpectedly includes a teacher')
        for module in system.student.modules():
            if hasattr(module, 'capture_transfer'):
                module.capture_transfer = False
                module.transfer_capture = None
        # Fixed-shape random input is appropriate for student latency, not training loss.
        image = torch.rand(1, 3, cfg['size'], cfg['size'], device=device)

        @torch.no_grad()
        def step(i, measure):
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            prediction = system.predict(image)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - begin) * 1000
            if not measure and not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite inference output')
            return elapsed

    for i in range(plan['warmup']):
        step(i, False)
    if world > 1:
        dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for i in range(plan['steps']):
        durations.append(step(i, True))
        if rank == 0 and (i + 1) % 25 == 0:
            print('%s %s repeat=%d step=%d/%d' % (args.role, args.phase, args.repeat + 1, i + 1, plan['steps']), flush=True)
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    result = dict(rank=rank, mean_ms=statistics.mean(durations), durations_ms=durations,
                  peak_allocated_bytes=peak, peak_reserved_bytes=reserved)
    if world > 1:
        ranks = [None] * world
        dist.all_gather_object(ranks, result)
    else:
        ranks = [result]
    if rank == 0:
        result = dict(phase=args.phase, role=args.role, repeat=args.repeat,
                      student_parameters=sum(p.numel() for p in system.student.parameters()),
                      environment=environment(), weight_load=load_report, ranks=ranks,
                      peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in ranks),
                      peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in ranks),
                      max_rank_mean_ms=max(r['mean_ms'] for r in ranks), mean_ms=ranks[0]['mean_ms'])
        write_json(args.result, result)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--dataset', choices=('isic17', 'isic18', 'bus', 'busi'), default='isic18')
    parser.add_argument('--seed', type=int, default=42)
    for role in ROLES:
        parser.add_argument('--run-' + role.lower(), default='')
    parser.add_argument('--data-root', default='')
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=0, help='0: use saved global batch size')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cached-batches', type=int, default=8)
    parser.add_argument('--output')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan', help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('train', 'infer'), help=argparse.SUPPRESS)
    parser.add_argument('--role', choices=ROLES, help=argparse.SUPPRESS)
    parser.add_argument('--repeat', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    if any(v < 1 for v in (args.gpus, args.warmup, args.steps, args.repeats, args.cached_batches)) or args.batch_size < 0:
        parser.error('GPU count, warmup, steps, repeats and cached batches must be positive')
    plan = prepare_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.output:
        parser.error('--output is required')
    output = Path(args.output).resolve()
    for record in plan['records'].values():
        if output == Path(record['run']) or Path(record['run']) in output.parents:
            raise ValueError('Output must not be inside a training run')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    for role in ROLES:
        write_json(output / ('config_%s.json' % role), plan['records'][role]['config'])
    # No inherited torchrun rank variables; this coordinator owns worker creation.
    env = dict(os.environ)
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID'):
        env.pop(key, None)
    try:
        for repeat in range(args.repeats):
            for role in ROLES:
                for phase in ('train', 'infer'):
                    command = [sys.executable]
                    if phase == 'train' and args.gpus > 1:
                        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                                    '--nproc_per_node=' + str(args.gpus)]
                    result = output / ('%s_%s_%d.json' % (phase, role, repeat))
                    command += [str(Path(__file__).resolve()), '--worker', '--plan', str(output / 'plan.json'),
                                '--phase', phase, '--role', role, '--repeat', str(repeat), '--result', str(result)]
                    print('Measuring %s %s (%d/%d)' % (role, phase, repeat + 1, args.repeats), flush=True)
                    subprocess.run(command, cwd=plan['project_root'], env=env, check=True)
        summarize(output, plan)
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='failed', error=str(exc)))
        raise
    print('Completed: ' + str(output / 'efficiency_table.md'), flush=True)


if __name__ == '__main__':
    main()
