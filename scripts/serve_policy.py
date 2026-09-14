import pathlib
import json
import dataclasses
import enum
import logging
import socket

import tyro

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
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

    # Semantic Action Guidance. **값의 truth 는 번들의 `cag_config.json`** 이다.
    #
    # 🔴 ω 는 번들의 속성이다. 24/30 은 "이 가중치 + ω=3" 조합에서 나온 값이라 둘을
    #    떼면 검증된 것이 아니다. 그래서 이 인자는 **덮어쓰지 않고 대조만** 한다:
    #
    #      안 주면      cag_config 의 guidance_scale 사용
    #      같으면       통과 (기동 로그에 한 줄 — 로그만 봐도 ω 를 알 수 있게)
    #      다르면       🔴 기동 거부
    #
    #    덮어쓰기를 막는 이유: ω 를 바꾸고 싶어지는 순간이 실기에서 실패했을 때인데,
    #    그때 명령줄 한 줄로 바뀌면 FREEZE 가 무의미하다. 바꾸려면 `cag_config.json` 을
    #    고쳐야 하고 그러면 번들 sha256 이 깨져 흔적이 남는다.
    cag_omega: float | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}



def _resolve_cag(bundle_dir: str, cli_omega: float | None) -> dict | None:
    """번들의 `cag_config.json` 이 truth. CLI 인자는 **대조만** 한다.

    반환: `sample_actions` 에 넘길 kwargs, 또는 CAG 를 안 쓰면 None.
    """
    cfg_path = pathlib.Path(bundle_dir) / "cag_config.json"
    if not cfg_path.exists():
        if cli_omega is not None:
            raise FileNotFoundError(
                f"--policy.cag-omega={cli_omega} 를 줬는데 {cfg_path} 가 없다. "
                "ω 는 번들의 속성이고, 파일 없이 값만 넘기면 무엇으로 검증된 조합인지 "
                "알 수 없다. 이 번들은 CAG 를 싣고 있지 않다."
            )
        logging.info("CAG 없음 (번들에 cag_config.json 없음) — 단일 브랜치")
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not cfg.get("enabled", False):
        logging.info("CAG 비활성 (cag_config.enabled=false) — 단일 브랜치")
        return None
    file_omega = float(cfg["guidance_scale"])

    if cli_omega is None:
        src = "cag_config"
    elif float(cli_omega) == file_omega:
        src = "cag_config (CLI 인자와 일치)"
    else:
        raise ValueError(
            f"OMEGA_MISMATCH — 기동 거부. --policy.cag-omega={cli_omega} 인데 "
            f"{cfg_path} 는 {file_omega} 다.\n"
            "  ω 는 번들의 속성이다. 검증된 24/30 은 '이 가중치 + 이 ω' 조합의 값이라\n"
            "  둘을 떼면 검증된 것이 아니다. CLI 로 덮어쓰지 않는다 — 바꾸려면\n"
            "  cag_config.json 을 고쳐야 하고, 그러면 번들 sha256 이 깨져 흔적이 남는다."
        )
    logging.info(
        f"CAG 활성 omega={file_omega} ({src}) form={cfg.get('form')} "
        f"steps={cfg.get('denoise_steps')} mix_at={cfg.get('mix_at')} "
        f"contract={cfg.get('contract_version')} — 관측마다 neutral_prompt 가 필요하다"
    )
    return {"cag_omega": file_omega}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            sample_kwargs = _resolve_cag(args.policy.dir, args.policy.cag_omega)
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir,
                default_prompt=args.default_prompt, sample_kwargs=sample_kwargs,
            )
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
