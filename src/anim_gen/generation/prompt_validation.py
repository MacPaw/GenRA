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

from ..config import PROMPT_VALIDATION_SYS_PROMPT_STR_RAW, get_system_config

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def validate_prompt_api(system_prompt_str: str, user_prompt_str: str, model: str = "gpt-4o") -> tuple[bool, str]:
    """
    Send a prompt validation request to the OpenAI API.

    Parameters:
        system_prompt_str (str): The system prompt string.
        user_prompt_str (str): The user's animation request prompt.
        model (str): The model to use for the request.

    Returns:
        tuple[bool, str]: A tuple containing (is_valid, reason).
                         is_valid is True if the prompt is valid, False otherwise.
                         reason is an empty string if valid, or a brief explanation if invalid.
    """
    logger.info("Validating prompt via OpenAI API...")
    logger.debug(f"User prompt: {user_prompt_str}")
    logger.debug(f"Model: {model}")

    messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt_str}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt_str}]},
    ]

    client = OpenAI()

    args = {
        "model": model,
        "input": messages,
        "text": {"format": {"type": "json_object"}},
        "service_tier": _system_config.service_tier.value,
    }

    # Use low reasoning effort for fast validation
    o_series_regex = r"^o[134]$"
    if (model.startswith("gpt-5") and "chat" not in model) or re.match(o_series_regex, model):
        args["reasoning"] = {"effort": "low", "summary": "detailed"}

    response = client.responses.create(**args)  # type: ignore

    response_str = response.output_text

    logger.debug(f"Tokens usage: {response.usage}")
    logger.debug(f"Validation response: {response_str}")

    if response_str is None or response_str.strip() == "":
        logger.warning("Validation response is empty, defaulting to valid")
        return True, ""

    response_str = response_str.strip()

    # Parse JSON response
    try:
        response_json = json.loads(response_str)

        # Validate JSON structure
        if not isinstance(response_json, dict):
            raise ValueError("Response is not a JSON object")

        if "pass" not in response_json:
            raise ValueError("Response missing 'pass' field")

        is_valid = bool(response_json["pass"])

        if is_valid:
            return True, ""
        else:
            # Extract reason if provided
            reason = response_json.get("reason", "")
            if not reason or not reason.strip():
                reason = "The requested animation is beyond the tool's capabilities."
            return False, reason.strip()

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # If JSON parsing fails, log warning and default to valid
        logger.warning(
            f"Failed to parse validation response as JSON: {e}. Response: {response_str}. Defaulting to valid."
        )
        return True, ""


def validate_prompt(user_prompt: str, model: str | None = None) -> tuple[bool, str]:
    """
    Validate if a user's animation prompt can be reliably generated with the tool.

    Parameters:
        user_prompt (str): The user's animation request prompt.
        model (str | None): The model to use. If None, uses system config default.

    Returns:
        tuple[bool, str]: A tuple containing (is_valid, reason).
                         is_valid is True if the prompt is valid, False otherwise.
                         reason is an empty string if valid, or a brief explanation if invalid.
    """
    if model is None:
        model = _system_config.prompt_validation_model

    system_prompt = PROMPT_VALIDATION_SYS_PROMPT_STR_RAW

    is_valid, reason = validate_prompt_api(system_prompt, user_prompt, model)

    if is_valid:
        logger.info("Prompt validation passed")
    else:
        logger.info(f"Prompt validation failed: {reason}")

    return is_valid, reason
