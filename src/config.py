"""Central configuration: paths, constants, and YAML config loading."""

from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_PSD_DIR = DATA_DIR / "raw" / "psd"
RAW_WASDE_DIR = DATA_DIR / "raw" / "wasde"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Ensure directories exist
for d in [RAW_PSD_DIR, RAW_WASDE_DIR, PROCESSED_DIR, FEATURES_DIR,
          EDA_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
SEED = 42


def load_commodities_config() -> dict:
    """Load the commodities YAML configuration."""
    config_path = CONFIGS_DIR / "commodities.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_commodity_codes(commodities: list[str] | None = None) -> dict[str, str]:
    """Return {name: code} mapping for requested commodities.

    Args:
        commodities: List of commodity names (e.g., ['wheat', 'corn']).
                     If None, returns all configured commodities.
    """
    config = load_commodities_config()
    all_commodities = config["commodities"]
    if commodities is None:
        commodities = list(all_commodities.keys())
    return {name: all_commodities[name]["code"] for name in commodities}


def get_attribute_ids(group: str = "balance_sheet") -> dict[str, str]:
    """Return {name: attribute_id} for the specified attribute group."""
    config = load_commodities_config()
    return config["attributes"][group]
