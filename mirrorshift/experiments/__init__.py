from mirrorshift.experiments.default import DEFAULT_TRAIN_SPEC
from mirrorshift.experiments.spec import TrainSpec

TRAIN_SPECS: dict[str, TrainSpec] = {
    DEFAULT_TRAIN_SPEC.name: DEFAULT_TRAIN_SPEC,
}


def get_train_spec(name: str) -> TrainSpec:
    if name not in TRAIN_SPECS:
        available = ", ".join(sorted(TRAIN_SPECS.keys()))
        raise ValueError(f"Unknown train spec '{name}'. Available specs: {available}")
    return TRAIN_SPECS[name]


def list_train_specs() -> list[str]:
    return sorted(TRAIN_SPECS.keys())
