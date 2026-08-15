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

import logging
import re

from openai import OpenAI

from ..config import (
    MOTION_DESCRIPTION_GENERATE_SYS_PROMPT_STR_RAW,
    MOTION_DESCRIPTION_REFINE_SYS_PROMPT_STR_RAW,
    get_system_config,
)
from .utils import response_usage_to_dict

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def optimize_motion_description_api(
    system_prompt_str: str, user_input_str: str, model: str = "gpt-5"
) -> tuple[str, dict | None]:
    """
    Send a motion description optimization request to the OpenAI API.

    Parameters:
        system_prompt_str (str): The system prompt string.
        user_input_str (str): The user input (prompt or refinement request).
        model (str): The model to use for the request.

    Returns:
        tuple[str, dict | None]: Optimized motion description and token usage dict (or None).
    """
    logger.info("Optimizing motion description via OpenAI API...")
    logger.debug(f"System prompt: {system_prompt_str}")
    logger.debug(f"User input: {user_input_str}")
    logger.debug(f"Model: {model}")

    messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt_str}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_input_str}]},
    ]

    client = OpenAI()

    args = {
        "model": model,
        "input": messages,
        "text": {"format": {"type": "text"}},
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

    # Clean up the response - remove any quotes if present
    response_str = response_str.strip().strip('"').strip("'")

    return str(response_str), token_usage


def optimize_motion_description_generate(user_prompt: str, model: str | None = None) -> tuple[str, dict | None]:
    """
    Optimize motion description for GENERATE mode.

    Parameters:
        user_prompt (str): The user's animation request prompt.
        model (str | None): The model to use. If None, uses system config default.

    Returns:
        tuple[str, dict | None]: Optimized motion description and token usage dict (or None).
    """
    if model is None:
        model = _system_config.motion_description_model

    system_prompt = MOTION_DESCRIPTION_GENERATE_SYS_PROMPT_STR_RAW

    optimized_description, token_usage = optimize_motion_description_api(system_prompt, user_prompt, model)

    logger.info(f"Optimized motion description (GENERATE): {optimized_description}")
    return optimized_description, token_usage


def optimize_motion_description_refine(
    base_motion_description: str, refinement_request: str, model: str | None = None
) -> tuple[str, dict | None]:
    """
    Optimize motion description for REFINE mode.

    Parameters:
        base_motion_description (str): The motion description from the base animation.
        refinement_request (str): The user's refinement request.
        model (str | None): The model to use. If None, uses system config default.

    Returns:
        tuple[str, dict | None]: Optimized motion description and token usage dict (or None).
    """
    if model is None:
        model = _system_config.motion_description_model

    system_prompt = MOTION_DESCRIPTION_REFINE_SYS_PROMPT_STR_RAW

    user_input = f"""Original motion description: "{base_motion_description}"

Refinement request: "{refinement_request}"

Please combine these into an updated motion description."""

    optimized_description, token_usage = optimize_motion_description_api(system_prompt, user_input, model)

    logger.info(f"Optimized motion description (REFINE): {optimized_description}")
    return optimized_description, token_usage
