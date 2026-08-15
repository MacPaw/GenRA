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
import os
import re
from pathlib import Path

from openai import OpenAI

from ..config import SELECTION_SYS_PROMPT_STR_RAW, get_system_config
from ..data_structs import AnimationFile
from .utils import response_usage_to_dict

logger = logging.getLogger(__name__)

_system_config = get_system_config()

ASSETS_PATH = Path(__file__).resolve().parent / "assets"


def select_example_api(system_prompt_str: str, user_request_str: str, model: str = "gpt-5") -> tuple[str, dict | None]:
    logger.info("Sending selection request to OpenAI API...")
    logger.debug(f"System prompt: {system_prompt_str}")
    logger.debug(f"Animation request: {user_request_str}")
    logger.debug(f"Model: {model}")

    messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt_str}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_request_str}]},
    ]

    client = OpenAI()

    args = {
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

    logger.debug(f"Tokens usage: {response.usage}")
    logger.debug(f"Output: {response.output}")

    if response_str is None or response_str.strip() == "":
        raise ValueError("Response from OpenAI API is empty or None.")

    return str(response_str), token_usage


def build_example_selection_prompt() -> str:
    if not os.path.exists(ASSETS_PATH / "selection_metadata.json"):
        raise FileNotFoundError(f"Metadata file not found at {ASSETS_PATH / 'selection_metadata.json'}")

    with open(ASSETS_PATH / "selection_metadata.json") as f:
        # load in plain text
        metadata_str = f.read()

    return SELECTION_SYS_PROMPT_STR_RAW.replace("{EXAMPLES_METADATA}", metadata_str)


def example_to_animation_file(example_name: str) -> AnimationFile:
    """
    Convert example name to AnimationFile.

    Parameters:
        example_name (str): Name of the internal example.

    Returns:
        AnimationFile: AnimationFile instance with path and caption.

    Raises:
        FileNotFoundError: If the example file or caption file doesn't exist.
    """

    def get_example_path(example_name: str) -> str:
        path = ASSETS_PATH / "usd_files" / f"{example_name}.usdc"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Example file not found at {path}")
        return str(path)

    def get_example_caption(example_name: str) -> dict:
        caption_path = ASSETS_PATH / "captions" / f"{example_name}.json"

        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found at {caption_path}")

        with open(caption_path) as f:
            caption = json.load(f)

        required_keys = ["model_description", "motion_description"]
        filtered_caption = {k: v for k, v in caption.items() if k in required_keys}

        return filtered_caption

    return AnimationFile(path=get_example_path(example_name), caption=get_example_caption(example_name))


def select_examples(user_request: str, model: str) -> tuple[list[AnimationFile], dict | None, list[str], str]:
    """
    Returns:
        tuple: (selected_examples, token_usage, selected_example_names, selection_reasoning).
    """
    prompt = build_example_selection_prompt()
    model_response, token_usage = select_example_api(prompt, user_request, model)

    try:
        model_response_json = json.loads(model_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from model: {model_response}") from e

    selection_reasoning = model_response_json.get("reasoning") or ""
    if selection_reasoning:
        logger.info(f"Selection reasoning: {selection_reasoning}")
    else:
        logger.warning("Model did not provide reasoning for the selection.")

    try:
        model_response_list = model_response_json["selected_examples"]
    except KeyError as e:
        raise ValueError(f"Invalid response from model: {model_response}") from e

    if not isinstance(model_response_list, list):
        raise ValueError(f"Invalid response from model: {model_response}")

    if not all(isinstance(item, str) for item in model_response_list):
        raise ValueError(f"Invalid response from model: {model_response}")

    # if len(model_response_list) < 1 or len(model_response_list) > 3:
    if len(model_response_list) > 3:
        raise ValueError(f"Invalid number of examples selected by model: {len(model_response_list)}")

    selected_examples = []

    for example_name in model_response_list:
        selected_examples.append(example_to_animation_file(example_name))

    logger.info(f"Selected {len(selected_examples)} examples for generation")
    logger.info(json.dumps([ex.to_dict() for ex in selected_examples], indent=4))

    return selected_examples, token_usage, model_response_list, selection_reasoning
