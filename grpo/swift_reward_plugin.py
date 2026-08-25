import os
import sys
import json
import time
import logging
import requests
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

_CHAPTERRL_ROOT = Path(__file__).resolve().parent.parent
if str(_CHAPTERRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTERRL_ROOT))

from swift.rewards import ORM, orms


class _RPCRewardBackend:
    def __init__(self, server_url: str, timeout: int = 300):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._healthy = None

    def health_check(self) -> bool:
        try:
            response = requests.get(f'{self.server_url}/health', timeout=5)
            if response.status_code != 200:
                self._healthy = False
                return False
            data = response.json()
            is_healthy = (
                data.get('status') == 'healthy'
                and data.get('calculator_initialized', False)
            )
            self._healthy = is_healthy
            if not is_healthy:
                logger.warning(f'[ChapterRL] Server unhealthy: {data}')
            return is_healthy
        except Exception as e:
            logger.warning(f'[ChapterRL] Health check failed: {e}')
            self._healthy = False
            return False

    def compute(self, completions: List[str], metadata: Optional[List[Dict]] = None) -> List[float]:
        if self._healthy is False:
            if not self.health_check():
                logger.error('[ChapterRL] Reward server not available')
                return self._fallback_rewards(len(completions))

        try:
            response = requests.post(
                f'{self.server_url}/compute',
                json={
                    'completions': completions,
                    'metadata': metadata,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                logger.debug(f'[ChapterRL] RPC rewards: mean={data["stats"]["mean"]:.3f}')
                return data['rewards']
            else:
                logger.error(f'[ChapterRL] RPC error: {response.status_code} - {response.text}')
                return self._fallback_rewards(len(completions))

        except requests.exceptions.Timeout:
            logger.error('[ChapterRL] RPC timeout')
            return self._fallback_rewards(len(completions))
        except Exception as e:
            logger.error(f'[ChapterRL] RPC failed: {e}')
            return self._fallback_rewards(len(completions))

    def _fallback_rewards(self, n: int) -> List[float]:
        return [-0.5] * n


class _LocalRewardBackend:
    _instance: Optional['_LocalRewardBackend'] = None

    def __init__(self):
        self._initialized = False
        self._calculator = None
        self._format_penalty = -0.5

    @classmethod
    def get(cls) -> '_LocalRewardBackend':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _lazy_init(self):
        if self._initialized:
            return

        root_dir = os.environ.get('CHAPTERRL_ROOT', str(_CHAPTERRL_ROOT))

        _legacy_base = os.environ.get('CHAPTERRL_BASE_MODEL', '')
        rdr_base_model = os.environ.get(
            'CHAPTERRL_RDR_BASE_MODEL',
            _legacy_base or '',
        )
        as_base_model = os.environ.get(
            'CHAPTERRL_AS_BASE_MODEL',
            _legacy_base or '',
        )

        rdr_ckpt = os.environ.get('CHAPTERRL_RDR_CHECKPOINT', '')
        as_ckpt = os.environ.get('CHAPTERRL_AS_CHECKPOINT', '')
        drt_enabled = os.environ.get('CHAPTERRL_DRT_ENABLED', 'true').lower() == 'true'
        format_penalty = float(os.environ.get('CHAPTERRL_FORMAT_PENALTY', '-0.5'))

        logger.info('[ChapterRL] Initializing local reward backend...')
        logger.info(f'  RDR Checkpoint  : {rdr_ckpt}')
        logger.info(f'  RDR Base Model  : {rdr_base_model}')
        logger.info(f'  AS  Checkpoint  : {as_ckpt}')
        logger.info(f'  AS  Base Model  : {as_base_model}')
        logger.info(f'  DRT Enabled     : {drt_enabled}')

        self._format_penalty = format_penalty

        try:
            from reward.unified_reward import UnifiedRewardCalculator
            self._calculator = UnifiedRewardCalculator(
                rdr_checkpoint=rdr_ckpt,
                rdr_base_model=rdr_base_model,
                as_checkpoint=as_ckpt if drt_enabled else None,
                as_base_model=as_base_model if drt_enabled else None,
                format_penalty=format_penalty,
                drt_enabled=drt_enabled,
                device='cpu',
            )
            logger.info('[ChapterRL] Local backend initialized')
        except Exception as e:
            logger.error(f'[ChapterRL] Failed to initialize local backend: {e}')
            import traceback
            traceback.print_exc()
            self._calculator = None

        self._initialized = True

    def compute(self, completions: List[str], metadata: Optional[List[Dict]] = None) -> List[float]:
        self._lazy_init()

        if self._calculator is None:
            logger.warning('[ChapterRL] Calculator not available')
            return [self._format_penalty] * len(completions)

        try:
            import torch
            rewards = self._calculator(completions, metadata)
            return rewards.tolist() if isinstance(rewards, torch.Tensor) else list(rewards)
        except Exception as e:
            logger.error(f'[ChapterRL] Reward computation failed: {e}')
            import traceback
            traceback.print_exc()
            return [self._format_penalty] * len(completions)


class ChapterRLORM(ORM):
    def __init__(self, args=None):
        super().__init__()

        mode = os.environ.get('CHAPTERRL_REWARD_MODE', 'rpc').lower()
        server_url = os.environ.get('CHAPTERRL_REWARD_SERVER', '')

        if mode == 'rpc':
            timeout = int(os.environ.get('CHAPTERRL_REWARD_TIMEOUT', '300'))
            self._backend = _RPCRewardBackend(server_url, timeout=timeout)
            logger.info(f'[ChapterRL] Using RPC mode, server: {server_url}, timeout: {timeout}s')
        else:
            self._backend = _LocalRewardBackend.get()
            logger.info('[ChapterRL] Using local mode (debug only)')

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        metadata = kwargs.get('metadata')
        if metadata is None:
            metadata = []
            for i in range(len(completions)):
                meta = {}
                for key in ['type', 'category', 'is_long']:
                    if key in kwargs:
                        val = kwargs[key]
                        if isinstance(val, list) and len(val) > i:
                            meta[key] = val[i]
                        elif not isinstance(val, list):
                            meta[key] = val
                metadata.append(meta if meta else None)

        rewards = self._backend.compute(completions, metadata)

        logger.debug(f'[ChapterRL] Computed {len(rewards)} rewards: '
                    f'mean={sum(rewards)/len(rewards):.3f}')

        if os.environ.get('CHAPTERRL_SAVE_COMPLETIONS', '0') == '1':
            self._save_completions(completions, rewards, metadata, kwargs)

        return rewards

    _step_counter: int = 0

    def _save_completions(
        self,
        completions: List[str],
        rewards: List[float],
        metadata: Optional[List],
        kwargs: dict,
    ) -> None:
        ChapterRLORM._step_counter += 1
        step = ChapterRLORM._step_counter

        interval = int(os.environ.get('CHAPTERRL_SAVE_INTERVAL', '1'))
        if step % interval != 0:
            return

        save_dir = os.environ.get('CHAPTERRL_COMPLETIONS_DIR', '')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'policy_outputs.jsonl')

        ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        records = []
        for i, (text, reward) in enumerate(zip(completions, rewards)):
            records.append({
                'step': step,
                'timestamp': ts,
                'sample_idx': i,
                'reward': round(reward, 6),
                'n_chars': len(text),
                'completion': text,
                'metadata': metadata[i] if metadata and i < len(metadata) else None,
            })

        with open(save_path, 'a', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        logger.info(
            f'[ChapterRL] Saved step={step} completions ({len(records)} samples) → {save_path}'
        )


orms['chapter_rl'] = ChapterRLORM
orms['chapterrl_unified'] = ChapterRLORM

logger.info('[ChapterRL] Reward plugin registered: chapter_rl, chapterrl_unified')
logger.info('[ChapterRL] Set CHAPTERRL_REWARD_MODE=rpc (recommended) or local (debug)')