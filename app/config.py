from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict
# from dotenv import load_dotenv

# Default configuration values used if no config.json or .env overrides are provided.
DEFAULTS: Dict[str, Any] = {
    "log": {
        "level": "INFO",               # Default logging level
        "to_file": False,              # Write logs to file? (default: only console)
        "file_path": "alerts.log",     # Log file location
        "file_max_bytes": 1_000_000,   # Max size of log file before rotation
        "file_backup_count": 3         # Number of rotated log files to keep
    },
    "state_file": "alert_state.json",  # File to persist alert state (anti-spam)
    "features_ridge_model": [          # Categorical feature columns for Ridge model training
        "MA3",
        "MA6",
        "MA12",
        "Lag_1",
        "Lag_3",
        "Lag_12",
        "pch_sm",
        "pch_sm_19",
        "NACEr2_Saison",
        "Land_Saison",
        "Aufenthaltsland_Saison",
        "Month_cycl_sin",
        "Month_cycl_cos",
        "Jahr",
        "pandemic_dummy"
    ],
      "country": {                      # default list of selected target countries with group classification
        "Deutschland": {
        "land_idx": "DE",
        "land_group": "Deutschland",
        "land_group_idx": "de"
        },
        "Spanien": {
        "land_idx": "ES",
        "land_group": "Mittelmeergroßezielländer",
        "land_group_idx": "medl"
        },
        "Portugal": {
        "land_idx": "PT",
        "land_group": "Mittelmeerkleinerezielländer",
        "land_group_idx": "meds"
        },
        "Italien": {
        "land_idx": "IT",
        "land_group": "Mittelmeergroßezielländer",
        "land_group_idx": "medl"
        },
        "Kroatien": {
        "land_idx": "HR",
        "land_group": "Mittelmeerkleinerezielländer",
        "land_group_idx": "meds"
        },
        "Griechenland": {
        "land_idx": "EL",
        "land_group": "Mittelmeerkleinerezielländer",
        "land_group_idx": "meds"
        },
        "Dänemark": {
        "land_idx": "DK",
        "land_group": "Nordischezielländer",
        "land_group_idx": "scandi"
        },
        "Norwegen": {
        "land_idx": "NR",
        "land_group": "Nordischezielländer",
        "land_group_idx": "scandi"
        },
        "Schweden": {
        "land_idx": "SW",
        "land_group": "Nordischezielländer",
        "land_group_idx": "scandi"
        },
        "Finnland": {
        "land_idx": "FI",
        "land_group": "Nordischezielländer",
        "land_group_idx": "scandi"
        }
    }

}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    """
    # Begin with a shallow copy of base
    out = dict(base)

    # Iterate through override items and merge/override accordingly
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v

    # Return the merged dictionary
    return out
 


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """
    Load the configuration for the application.

    Priority:
    1. Default values from DEFAULTS
    2. Overrides from config.json (if present)
    3. Overrides from environment variables (.env or OS-level)
    """
    # Load environment variables via load_dotenv()
    # load_dotenv()

    # Read config.json (if it exists) and parse JSON into 'user'
    user = {}
    p = Path(path)
    if p.exists():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"config.json could not be read: {e}")

    # Merge DEFAULTS with user config using deep_merge()
    cfg = deep_merge(base=DEFAULTS, override=user)

    # Apply environment variable overrides (LOG_LEVEL,...)
    if os.getenv("LOG_LEVEL"):
        cfg["log"]["level"] = os.getenv("LOG_LEVEL")

    
    # Validate critical settings (ntfy topic, tickers)
    if not cfg["country"]["Deutschland"]["land_idx"]  == "DE":
        raise RuntimeError(
            """
            Please set a aminimum 'Deutschland (DE)" as a country in config.json or .env
            """
            )
    
    
    # Return the final configuration dictionary
    return cfg

