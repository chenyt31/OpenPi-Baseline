from collections.abc import Sequence
import dataclasses
import enum
import logging
import socket
from typing import Any

import tyro

from openpi import transforms
from openpi.models import exported as _exported
from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Exported:
    """Load an exported checkpoint."""

    # Checkpoint directory (e.g., "s3://openpi-assets/exported/pi0_base/model").
    dir: str
    # Processor name to load the norm stats from. If not provided, will automatically load a processor if there is only
    # one available. If there are multiple processors, raise an error and ask the user to provide a processor name.
    processor: str | None = None


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies, or loading exported models.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, overrides the default prompt for the policy.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    # TODO(ury): Remove support for exported models before releasing.
    policy: Checkpoint | Exported | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    # TODO(ury): Update to use the base checkpoint.
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha_towel",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_towel",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    # TODO(ury): Make sure that this works once the checkpoint is ready.
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_exported_policy(env: EnvMode, exported: Exported, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a policy from an exported model."""
    checkpoint_dir = exported.dir
    processor = exported.processor

    logging.info("Loading model...")
    model = _exported.PiModel(checkpoint_dir)

    processors = model.processor_names()
    if not processors:
        raise ValueError(f"No processors found in {checkpoint_dir}")

    if processor is None:
        # If there is only one processor, use it.
        if len(processors) == 1:
            processor = processors[0]
        # If there are multiple processors, ask the user to provide a processor name.
        else:
            raise ValueError(f"Processor name must be provided. Available: {processors}")
        assert processor is not None
        logging.info("Using processor: %s", processor)
    elif processor not in processors:
        raise ValueError(f"Processor {processor} not found in {checkpoint_dir}, found {processors}")

    def make_policy_config(
        input_layers: Sequence[transforms.DataTransformFn],
        output_layers: Sequence[transforms.DataTransformFn],
        sample_kwargs: dict[str, Any] | None = None,
    ):
        return _policy_config.PolicyConfig(
            model=model,
            norm_stats=model.norm_stats(processor),
            default_prompt=default_prompt,
            input_layers=input_layers,
            output_layers=output_layers,
            model_type=model.model_type,
            sample_kwargs=sample_kwargs,
        )

    logging.info("Creating policy...")
    match env:
        case EnvMode.ALOHA:
            delta_action_mask = transforms.make_bool_mask(6, -1, 6, -1)
            config = make_policy_config(
                input_layers=[
                    aloha_policy.AlohaInputs(action_dim=model.action_dim),
                    transforms.DeltaActions(mask=delta_action_mask),
                ],
                output_layers=[
                    transforms.AbsoluteActions(mask=delta_action_mask),
                    aloha_policy.AlohaOutputs(),
                ],
            )
        case EnvMode.ALOHA_SIM:
            config = make_policy_config(
                input_layers=[aloha_policy.AlohaInputs(action_dim=model.action_dim)],
                output_layers=[aloha_policy.AlohaOutputs()],
            )
        case EnvMode.DROID:
            config = make_policy_config(
                input_layers=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=model.model_type)],
                output_layers=[droid_policy.DroidOutputs()],
            )
        case EnvMode.LIBERO:
            config = make_policy_config(
                input_layers=[libero_policy.LiberoInputs(action_dim=model.action_dim)],
                output_layers=[libero_policy.LiberoOutputs()],
            )
        case _:
            raise ValueError(f"Unknown environment mode: {env}")

    return _policy_config.create_policy(config)


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Exported():
            return create_exported_policy(args.env, args.policy, default_prompt=args.default_prompt)
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
