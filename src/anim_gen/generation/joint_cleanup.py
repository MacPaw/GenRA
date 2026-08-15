# Copyright 2026 MacPaw Way Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License") with the
# "Commons Clause" License Condition v1.0; you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# and the Commons Clause condition in the LICENSE file distributed with this
# software.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re

from openai import OpenAI

from ..config import JOINT_CLEANUP_SYS_PROMPT_STR_RAW, get_system_config
from .utils import response_usage_to_dict

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def _cleanup_joint_names_api(
    system_prompt_str: str, user_input_str: str, model: str = "gpt-5.4"
) -> tuple[str, dict | None]:
    """Raw LLM call that returns the model response and token usage."""
    logger.info("Sending joint-name cleanup request to OpenAI API...")
    logger.debug("Model: %s", model)

    messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt_str}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_input_str}]},
    ]

    client = OpenAI()

    args: dict = {
        "model": model,
        "input": messages,
        "text": {"format": {"type": "json_object"}},
        "service_tier": _system_config.service_tier.value,
    }

    o_series_regex = r"^o[134]$"
    if (model.startswith("gpt-5") and "chat" not in model) or re.match(o_series_regex, model):
        args["reasoning"] = {"effort": "low", "summary": "detailed"}

    response = client.responses.create(**args)  # type: ignore

    response_str = response.output_text
    token_usage = response_usage_to_dict(getattr(response, "usage", None))

    logger.debug("Cleanup tokens usage: %s", response.usage)

    if response_str is None or response_str.strip() == "":
        raise ValueError("Joint-name cleanup response from OpenAI API is empty.")

    return str(response_str), token_usage


def _extract_leaf_names(joint_paths: list[str]) -> list[str]:
    """Extract the leaf (last segment after '/') from each joint path, preserving order."""
    return [p.rsplit("/", 1)[-1] for p in joint_paths]


def _validate_mapping(leaf_names: list[str], mapping: dict[str, str]) -> tuple[bool, str]:
    """Validate that the mapping is complete, 1:1, and all values are non-empty and unique."""
    leaf_set = set(leaf_names)

    missing = leaf_set - mapping.keys()
    if missing:
        return False, f"missing entries for: {sorted(missing)[:5]}"

    extra = mapping.keys() - leaf_set
    if extra:
        return False, f"unexpected entries: {sorted(extra)[:5]}"

    for old, new in mapping.items():
        if not new or not new.strip():
            return False, f"empty output for '{old}'"

    values = list(mapping.values())
    if len(values) != len(set(values)):
        dupes = [v for v in values if values.count(v) > 1]
        return False, f"duplicate outputs: {sorted(set(dupes))[:5]}"

    return True, ""


def cleanup_joint_names(
    joint_paths: list[str], model: str | None = None
) -> tuple[dict[str, str], dict[str, str], dict | None]:
    """Produce a forward and reverse leaf-name mapping via LLM cleanup.

    Returns:
        (forward_map, reverse_map, token_usage)
        Maps are {original_leaf: cleaned_leaf} and {cleaned_leaf: original_leaf}.
        If the LLM returns identity or validation fails, maps are identity.
    """
    if model is None:
        model = _system_config.cleanup_model

    leaf_names = _extract_leaf_names(joint_paths)
    unique_leaves = sorted(set(leaf_names))

    user_input = json.dumps(unique_leaves)
    logger.debug("Joint cleanup input (%d unique leaves): %s", len(unique_leaves), user_input)

    try:
        response_str, token_usage = _cleanup_joint_names_api(JOINT_CLEANUP_SYS_PROMPT_STR_RAW, user_input, model)
    except Exception:
        logger.warning("Joint-name cleanup LLM call failed; falling back to identity", exc_info=True)
        identity = {n: n for n in unique_leaves}
        return identity, identity, None

    try:
        response_json = json.loads(response_str)
        mapping: dict[str, str] = response_json["mapping"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse cleanup response (%s); falling back to identity", exc)
        identity = {n: n for n in unique_leaves}
        return identity, identity, token_usage

    valid, reason = _validate_mapping(unique_leaves, mapping)
    if not valid:
        logger.warning("Cleanup mapping validation failed (%s); falling back to identity", reason)
        identity = {n: n for n in unique_leaves}
        return identity, identity, token_usage

    is_identity = all(k == v for k, v in mapping.items())
    if is_identity:
        logger.info("Joint names already clean, no renaming needed")
        return mapping, {v: k for k, v in mapping.items()}, token_usage

    renamed_count = sum(1 for k, v in mapping.items() if k != v)
    sample = {k: v for i, (k, v) in enumerate(mapping.items()) if k != v and i < 5}
    logger.info("Renaming %d/%d joint names, sample: %s", renamed_count, len(mapping), sample)
    logger.debug("Full cleanup mapping: %s", mapping)

    forward_map = dict(mapping)
    reverse_map = {v: k for k, v in mapping.items()}
    return forward_map, reverse_map, token_usage


def apply_joint_name_mapping(joint_paths: list[str], leaf_map: dict[str, str]) -> list[str]:
    """Apply a leaf-name mapping to full joint paths (replace each path segment)."""
    result = []
    for path in joint_paths:
        segments = path.split("/")
        mapped = "/".join(leaf_map.get(seg, seg) for seg in segments)
        result.append(mapped)
    return result


def rename_json_keys(json_dict: dict, full_path_map: dict[str, str]) -> dict:
    """Replace top-level keys using a full-path-to-full-path mapping. Preserves value order."""
    return {full_path_map.get(k, k): v for k, v in json_dict.items()}
