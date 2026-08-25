import os
os.environ.setdefault("DS_ACCELERATOR", "cuda")
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import sys
import argparse
import logging
import traceback
from typing import Optional

_CHAPTERRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHAPTERRL_ROOT not in sys.path:
    sys.path.insert(0, _CHAPTERRL_ROOT)

from flask import Flask, request, jsonify
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_calculator = None
_config = {}
_init_error: Optional[str] = None


def _str_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    v_lower = str(v).strip().lower()
    if v_lower in ('true', 't', 'yes', 'y', '1', 'on', 'enabled'):
        return True
    if v_lower in ('false', 'f', 'no', 'n', '0', 'off', 'disabled'):
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected (true/false/yes/no/1/0), got: {v!r}"
    )


def _detect_device() -> str:
    env_dev = os.environ.get('CHAPTERRL_REWARD_DEVICE')
    if env_dev:
        return env_dev
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_calculator():
    global _calculator, _init_error

    if _calculator is not None:
        return _calculator

    if _init_error is not None:
        raise RuntimeError(f"Calculator initialization previously failed: {_init_error}")

    if not _config.get('rdr_checkpoint') or not _config.get('rdr_base_model'):
        raise RuntimeError(
            "Cannot initialize calculator: --rdr_checkpoint and --rdr_base_model are required."
        )

    device = _detect_device()
    logger.info('[RewardServer] Initializing UnifiedRewardCalculator...')
    logger.info(f'  Device          : {device}')
    logger.info(f'  RDR Checkpoint  : {_config["rdr_checkpoint"]}')
    logger.info(f'  RDR Base Model  : {_config["rdr_base_model"]}')
    logger.info(f'  AS  Checkpoint  : {_config.get("as_checkpoint")}')
    logger.info(f'  AS  Base Model  : {_config.get("as_base_model")}')
    logger.info(f'  DRT Enabled     : {_config["drt_enabled"]}')

    try:
        from reward.unified_reward import UnifiedRewardCalculator

        _calculator = UnifiedRewardCalculator(
            rdr_checkpoint=_config['rdr_checkpoint'],
            rdr_base_model=_config['rdr_base_model'],
            as_checkpoint=_config.get('as_checkpoint'),
            as_base_model=_config.get('as_base_model'),
            format_penalty=_config.get('format_penalty', -0.5),
            drt_enabled=_config['drt_enabled'],
            device=device,
        )

        if _calculator.rdr_infer is None:
            raise RuntimeError("RDR model failed to load (rdr_infer is None)")

        if _config['drt_enabled'] and _calculator.as_model is None:
            logger.warning(
                "[RewardServer] DRT was requested but AS model failed to load. "
                "Calculator will run in RDR-only fallback mode."
            )

        logger.info('[RewardServer] UnifiedRewardCalculator ready')
        return _calculator

    except Exception as e:
        _init_error = str(e)
        logger.error(f'[RewardServer] FATAL: Failed to initialize calculator: {e}')
        traceback.print_exc()
        raise


@app.route('/health', methods=['GET'])
def health_check():
    ready = (_calculator is not None) and (_init_error is None)
    return jsonify({
        'status': 'healthy',
        'calculator_initialized': _calculator is not None,
        'ready_for_compute': ready,
        'init_error': _init_error,
        'config': {
            'rdr_checkpoint': _config.get('rdr_checkpoint'),
            'rdr_base_model': _config.get('rdr_base_model'),
            'as_checkpoint': _config.get('as_checkpoint'),
            'as_base_model': _config.get('as_base_model'),
            'drt_enabled': _config.get('drt_enabled'),
            'format_penalty': _config.get('format_penalty'),
        },
    })


@app.route('/compute', methods=['POST'])
def compute_rewards():
    try:
        data = request.get_json()
        if not data or 'completions' not in data:
            return jsonify({'error': 'Missing field: completions'}), 400

        completions = data['completions']
        metadata = data.get('metadata')

        if not isinstance(completions, list) or len(completions) == 0:
            return jsonify({'error': 'completions must be a non-empty list'}), 400

        logger.info(f'[RewardServer] Computing rewards for {len(completions)} samples')

        calc = get_calculator()
        rewards_tensor = calc(completions, metadata, verbose=True)

        rewards = (
            rewards_tensor.tolist()
            if isinstance(rewards_tensor, torch.Tensor)
            else list(rewards_tensor)
        )

        arr = np.array(rewards, dtype=np.float64)
        format_penalty_val = float(_config.get('format_penalty', -0.5))
        n_format = int(np.sum(np.isclose(arr, format_penalty_val, rtol=1e-5, atol=1e-6)))

        stats = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'n_format_penalty': n_format,
            'format_penalty_ratio': round(n_format / max(len(rewards), 1), 3),
        }
        logger.info(
            f'[RewardServer] Done — mean={stats["mean"]:.3f}, std={stats["std"]:.3f}, '
            f'min={stats["min"]:.3f}, max={stats["max"]:.3f}, '
            f'format_penalty={n_format}/{len(rewards)} ({stats["format_penalty_ratio"]*100:.1f}%)'
        )

        return jsonify({'rewards': rewards, 'stats': stats})

    except RuntimeError as e:
        logger.error(f'[RewardServer] RuntimeError: {e}')
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(f'[RewardServer] Error: {e}')
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    logger.info('[RewardServer] Shutdown requested')
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return jsonify({'status': 'shutting down'})


def main():
    parser = argparse.ArgumentParser(
        description='ChapterRL Reward Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--port', type=int, default=29601)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--rdr_checkpoint', type=str, default=None)
    parser.add_argument('--rdr_base_model', type=str, default=None)
    parser.add_argument('--as_checkpoint', type=str, default=None)
    parser.add_argument('--as_base_model', type=str, default=None)
    parser.add_argument('--drt_enabled', type=_str_to_bool, default=True)
    parser.add_argument('--format_penalty', type=float, default=-0.5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--base_model', type=str, default=None)

    args = parser.parse_args()

    if args.base_model:
        if not args.rdr_base_model:
            args.rdr_base_model = args.base_model
        if not args.as_base_model:
            args.as_base_model = args.base_model

    if not args.rdr_checkpoint or not args.rdr_base_model:
        parser.error("--rdr_checkpoint and --rdr_base_model are required")

    if args.drt_enabled:
        if not args.as_checkpoint or not args.as_base_model:
            logger.warning(
                "[RewardServer] --drt_enabled=true but --as_checkpoint or "
                "--as_base_model is missing. Falling back to RDR-only mode."
            )
            args.drt_enabled = False

    global _config
    _config = {
        'rdr_checkpoint': args.rdr_checkpoint,
        'rdr_base_model': args.rdr_base_model,
        'as_checkpoint': args.as_checkpoint,
        'as_base_model': args.as_base_model,
        'drt_enabled': args.drt_enabled,
        'format_penalty': args.format_penalty,
    }

    logger.info('=' * 60)
    logger.info('ChapterRL Reward Server')
    logger.info('=' * 60)
    logger.info(f'  Host            : {args.host}:{args.port}')
    logger.info(f'  Device          : {_detect_device()}')
    logger.info(f'  CUDA available  : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            logger.info(f'  GPU {i:<2}          : '
                        f'{torch.cuda.get_device_name(i)} '
                        f'({free/1024**3:.1f}/{total/1024**3:.1f} GB free)')
    logger.info(f'  RDR Checkpoint  : {args.rdr_checkpoint}')
    logger.info(f'  RDR Base Model  : {args.rdr_base_model}')
    logger.info(f'  AS  Checkpoint  : {args.as_checkpoint}')
    logger.info(f'  AS  Base Model  : {args.as_base_model}')
    logger.info(f'  DRT Enabled     : {args.drt_enabled}')
    logger.info(f'  Format Penalty  : {args.format_penalty}')
    logger.info(f'  DS_ACCELERATOR  : {os.environ.get("DS_ACCELERATOR")}')
    logger.info('=' * 60)

    logger.info('[RewardServer] Pre-loading models (this may take a few minutes) ...')
    try:
        get_calculator()
        logger.info('[RewardServer] ✓ Models pre-loaded successfully')
    except Exception as e:
        logger.error(f'[RewardServer] ✗ Pre-load failed: {e}')
        logger.error('[RewardServer] Server will start anyway but /compute will return 503')

    logger.info(f'[RewardServer] Listening on {args.host}:{args.port}')
    logger.info('[RewardServer] Endpoints:')
    logger.info(f'  GET  http://{args.host}:{args.port}/health    - 健康检查')
    logger.info(f'  POST http://{args.host}:{args.port}/compute   - 计算奖励')
    logger.info(f'  POST http://{args.host}:{args.port}/shutdown  - 关闭服务')

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()