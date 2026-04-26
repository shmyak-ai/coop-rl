"""Runtime orchestration for Ray-based and local-thread training."""

import logging
import tempfile
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from coop_rl.configs import get_config

RUNTIME_ENV_CPU = {
    "env_vars": {
        "JAX_PLATFORMS": "cpu",
    }
}

RUNTIME_ENV_GPU = {
    "env_vars": {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "RAY_DEDUP_LOGS": "0",
    }
}

RUNTIME_ENV_DEBUG = {
    "env_vars": {
        "RAY_DEBUG": "1",
        "RAY_DEBUG_POST_MORTEM": "1",
        "RAY_DEDUP_LOGS": "0",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_DISABLE_JIT": "true",
    }
}


def create_workdir(workdir_root: str) -> str:
    """Create a dedicated run directory under the configured root."""
    base_dir = Path(workdir_root).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="run-", dir=base_dir)


def load_runtime_config(config_name: str, checkpoint_dir: str | None) -> Any:
    """Build and finalize the runtime config."""
    conf = get_config(config_name)
    conf.observation_shape, conf.observation_dtype, conf.actions_shape = conf.env.check_env(
        **conf.args_env
    )
    conf.args_state_recover.checkpointdir = checkpoint_dir
    return conf


def configure_logging(conf: Any, debug_log: bool) -> logging.Logger:
    """Configure runtime logging and propagate log level into the config."""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    log_level = "DEBUG" if debug_log else "INFO"
    logger.setLevel(log_level)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    conf.log_level = log_level
    return logger


def decorate_remote_components(conf: Any) -> Any:
    """Convert runtime classes into Ray remote actors."""
    import ray

    conf.controller = ray.remote(
        num_cpus=0,
        num_gpus=0,
        runtime_env=RUNTIME_ENV_CPU,
    )(conf.controller)
    conf.trainer = ray.remote(num_cpus=1, num_gpus=0.5, runtime_env=RUNTIME_ENV_GPU)(conf.trainer)
    conf.collector = ray.remote(
        num_cpus=1,
        num_gpus=0.5 / conf.num_collectors,
        runtime_env=RUNTIME_ENV_GPU,
    )(conf.collector)
    return conf


def launch_remote_workers(conf: Any) -> tuple[Any, list[Any]]:
    """Launch controller, trainer, and collector actors."""
    controller = conf.controller.remote()
    trainer = conf.trainer.options(max_concurrency=3 + conf.num_samplers).remote(
        **conf.args_trainer, controller=controller
    )

    collectors = []
    for _ in range(conf.num_collectors):
        conf.args_collector.collectors_seed += 1
        collector = conf.collector.remote(
            **conf.args_collector,
            controller=controller,
            trainer=trainer,
        )
        collectors.append(collector)

    return trainer, collectors


def initialize_ray(debug_ray: bool) -> None:
    """Initialize Ray with optional debugging runtime settings."""
    import ray

    if debug_ray:
        ray.init(runtime_env=RUNTIME_ENV_DEBUG)
    else:
        ray.init()


def run_training(args: Namespace) -> None:
    """Execute training from parsed CLI arguments."""
    logger = logging.getLogger(__name__)

    conf = load_runtime_config(args.config, args.orbax_checkpoint_dir)
    logger = configure_logging(conf, args.debug_log)
    conf.workdir = create_workdir(args.workdir)
    logger.info("Workdir is %s.", conf.workdir)

    if args.backend == "thread":
        _run_thread_training(conf)
        return

    import ray

    initialize_ray(args.debug_ray)

    try:
        conf = decorate_remote_components(conf)
        trainer, collectors = launch_remote_workers(conf)

        trainer_futures = (
            [trainer.training.remote(), trainer.buffer_updating.remote()]
            + [trainer.buffer_sampling.remote() for _ in range(conf.num_samplers)]
            + [trainer.add_traj_seq.remote(1)]
        )
        collect_info_futures = [collector.collecting.remote() for collector in collectors]

        ray.get(trainer_futures)
        ray.get(collect_info_futures)
        time.sleep(3)
    finally:
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Done; ray shutdown.")


def _launch_thread_workers(conf: Any) -> tuple[Any, Any, list[Any]]:
    """Launch controller, trainer, and collectors as local objects."""
    conf.args_trainer.controller = controller = conf.controller()
    trainer = conf.trainer(**conf.args_trainer)

    collectors = []
    conf.args_collector.controller = controller
    conf.args_collector.trainer = trainer
    for _ in range(conf.num_collectors):
        conf.args_collector.collectors_seed += 1
        collector = conf.collector(**conf.args_collector)
        collectors.append(collector)

    return controller, trainer, collectors


def _run_thread_training(conf: Any) -> None:
    """Execute training loop with regular Python multithreading."""
    controller, trainer, collectors = _launch_thread_workers(conf)
    max_workers = 3 + conf.num_samplers + conf.num_collectors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(trainer.training),
            executor.submit(trainer.buffer_updating),
            *[executor.submit(trainer.buffer_sampling) for _ in range(conf.num_samplers)],
            executor.submit(trainer.add_traj_seq, 1),
            *[executor.submit(collector.collecting) for collector in collectors],
        ]

        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            # Unblock all worker loops when one thread fails.
            trainer.is_done = True
            controller.set_done()
            raise

    time.sleep(3)
