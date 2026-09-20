"""Load and resolve per-framework fine-tuning settings from finetune/settings/<Framework>.json.

Precedence: user value > official fine-tuning value (ft_value) > code default.
A required setting with no default and no user value is an error, except those
the builder itself fills (dataset paths, foundation checkpoint).

Returns per setting: value and origin ("user" | "official-finetune" | "default" | "builder").
Common knobs map to native settings through `knob` in the settings JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent

# Common knob names recognized across frameworks
COMMON_KNOBS = {
    "epochs", "max_steps", "batch_size", "lr", "scheduler", "energy_weight",
    "force_weight", "stress_weight", "include_stress", "val_fraction", "seed",
    "patience", "ema", "precision", "device", "num_workers", "head", "freeze", "cutoff", "max_time"
}


class SettingsError(Exception):
    pass


def load_settings_file(framework: str) -> dict:
    """Load finetune/settings/<Framework>.json from the repo root."""
    settings_path = _REPO_ROOT / "finetune" / "settings" / f"{framework}.json"
    try:
        with open(settings_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SettingsError(f"Settings file not found: {settings_path}")
    except json.JSONDecodeError as e:
        raise SettingsError(f"Invalid JSON in {settings_path}: {e}")


def _parse_yaml_scalar(value_str: str) -> Any:
    """Parse a YAML scalar value from CLI --set NAME=VALUE."""
    s = value_str.strip()
    if not s:
        return ""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "none"):
        return None
    # Try to parse as number
    try:
        # Try float first (handles both "3.14" and "1e-4")
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s  # return as string


def resolve_settings(
    framework: str,
    user_values: dict[str, Any] | None = None,
    user_knobs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve framework settings with precedence: user > ft_value > default.

    Only returns settings that have an actual value. Settings with no value
    (internal/builder-filled) are omitted from the result.

    Args:
        framework: Framework name (e.g., "MACE")
        user_values: Native setting names -> values (from --set NAME=VALUE)
        user_knobs: Knob names -> values (from CLI flags like --epochs, --lr)

    Returns:
        (settings_dict, origins_dict) where:
        - settings_dict maps native names to resolved values
        - origins_dict maps native names to origin ("user", "official-finetune", "default")
    """
    if user_values is None:
        user_values = {}
    if user_knobs is None:
        user_knobs = {}

    settings_file = load_settings_file(framework)
    settings_list = settings_file.get("settings", [])

    resolved = {}
    origins = {}

    for setting_entry in settings_list:
        native_name = setting_entry["name"]
        knob = setting_entry.get("knob")
        default_val = setting_entry.get("default")
        ft_val = setting_entry.get("ft_value")

        # Determine which value to use based on precedence
        final_value = None
        origin = None

        # Priority 1: user --set NAME=VALUE
        if native_name in user_values:
            final_value = user_values[native_name]
            origin = "user"
        # Priority 2: user knob flag (e.g., --epochs)
        elif knob and knob in user_knobs:
            final_value = user_knobs[knob]
            origin = "user"
        # Priority 3: official fine-tuning value
        elif ft_val is not None:
            final_value = ft_val
            origin = "official-finetune"
        # Priority 4: code default
        elif default_val is not None:
            final_value = default_val
            origin = "default"
        else:
            # No value available - skip this setting
            # (it's either internal, built by the builder, or filled by prestage)
            continue

        # Numbers written as text in the settings data ("1e-8", "0.00") are numbers
        # upstream; emit them as numbers unless the setting is a string setting.
        if isinstance(final_value, str) and "str" not in str(setting_entry.get("type") or "").lower():
            try:
                num = float(final_value)
            except ValueError:
                pass
            else:
                final_value = int(num) if num.is_integer() and not any(c in final_value for c in ".eE") else num
        # Store the resolved value and origin
        resolved[native_name] = final_value
        origins[native_name] = origin

    return resolved, origins


def show_settings(framework: str) -> str:
    """Print the framework's settings table for LLM agent inspection.

    Shows each native setting with its knob mapping (if any), defaults, and
    official fine-tuning values. For lookups, knob names map to the native
    setting name shown in the Native Name column.
    """
    settings_file = load_settings_file(framework)
    settings_list = settings_file.get("settings", [])

    lines = [
        f"# Settings for {framework}",
        f"Pinned: {settings_file.get('pinned', 'N/A')}",
        "",
        "| Native Name | Knob → Native | Default | Official FT | Type | Category | Required |",
        "|---|---|---|---|---|---|---|",
    ]

    for s in settings_list:
        native = s["name"]
        knob = s.get("knob")
        # Show knob-to-native mapping if a knob exists; a setting that is only related to a
        # knob's topic is not set by that flag and is reachable with --set
        if knob:
            knob_mapping = f"{knob} → {native}"
        elif s.get("related_knob"):
            knob_mapping = f"(related to {s['related_knob']}; use --set)"
        else:
            knob_mapping = "—"
        default = s.get("default", "—")
        ft_val = s.get("ft_value") or "—"
        stype = s.get("type", "?")
        category = s.get("category", "?")
        req = "yes" if s.get("default_required") else "no"

        lines.append(f"| {native} | {knob_mapping} | {default} | {ft_val} | {stype} | {category} | {req} |")

    return "\n".join(lines)


def validate_setting_names(framework: str, names: set[str]) -> None:
    """Raise SettingsError for a --set NAME the framework does not declare.

    An unknown name used to be dropped in silence: resolve_settings only walks
    the declared settings, so `--set bogus=1` (or a typo in a real name) never
    reached the trainer and never said so. docs/howto/finetune.md advertises
    --set NAME=VALUE for any native setting, which makes silence the worst
    answer -- the user believes the value was applied."""
    settings_file = load_settings_file(framework)
    declared = {s["name"] for s in settings_file.get("settings", []) if s.get("name")}
    unknown = {n for n in names if n not in declared}
    if not unknown:
        return
    import difflib

    lines = []
    for name in sorted(unknown):
        close = difflib.get_close_matches(name, sorted(declared), n=3, cutoff=0.6)
        hint = f" -- did you mean {', '.join(close)}?" if close else ""
        lines.append(f"  {name}{hint}")
    raise SettingsError(
        f"[ft_settings] {framework} does not declare these settings:\n"
        + "\n".join(lines)
        + f"\n  Run `python3 scripts/ft_run.py {framework} --show-settings` for the list."
    )


def validate_knobs_exist(framework: str, knob_names: set[str]) -> None:
    """Raise SettingsError if any knob is not exposed by the framework."""
    settings_file = load_settings_file(framework)
    settings_list = settings_file.get("settings", [])
    available_knobs = {s.get("knob") for s in settings_list if s.get("knob")}

    unsupported = knob_names - available_knobs
    if unsupported:
        unsupported_str = ", ".join(sorted(unsupported))
        available_str = ", ".join(sorted(available_knobs - {None})) or "none"
        raise SettingsError(
            f"[ft_settings] {framework} does not expose these knobs: {unsupported_str}\n"
            f"  Available knobs: {available_str}"
        )
