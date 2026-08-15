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

import base64
import copy
import logging
import pprint
import re
from typing import Any

from openai import OpenAI

from ..config import RIG_RENDER_USER_PROMPT, GenerationMode, get_system_config
from .utils import response_usage_to_dict

logger = logging.getLogger(__name__)

_system_config = get_system_config()


_BASE64_PREFIX = "data:image/"


def _encode_image_base64(image_path: str) -> str:
    """Encode a local image file as a base64 data URL for the OpenAI API."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _sanitize_messages_for_log(messages: list[dict]) -> list[dict]:
    """Return a deep copy of messages with base64 image data replaced by a placeholder."""
    sanitized = copy.deepcopy(messages)
    for msg in sanitized:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            url = item.get("image_url", "")
            if isinstance(url, str) and url.startswith(_BASE64_PREFIX):
                item["image_url"] = f"{url[:30]}...<{len(url)} chars>"
    return sanitized


def api_request(
    system_prompt_str: str,
    user_prompt_str_list: list[str],
    user_request_str: str,
    image_paths: list[str] | None = None,
    use_code_interpreter: bool = False,
    model: str = "gpt-4o",
    temperature: float = 1.0,
    top_p: float = 1.0,
    mode: GenerationMode = GenerationMode.GENERATE,
    reasoning_effort: str = "medium",
) -> tuple[str, list[dict], dict | None]:
    """
    Send a generation request to the OpenAI API and return the response.

    Parameters:
        system_prompt_str (str): The system prompt string.
        user_prompt_str_list (list[str]): A list of user prompt strings.
        user_request_str (str): The user request string.
        image_paths (list[str] | None): Optional list of local image file paths to include
            as a user message (e.g. rig render views). Inserted after text prompts, before user request.
        use_code_interpreter (bool): Whether to enable the Code Interpreter tool, allowing
            the model to run Python code during generation.
        model (str): The model to use for the request (default is "gpt-4o").
        temperature (float): The temperature for the generation (default is 1.0).
        top_p (float): The top_p for the generation (default is 1.0).
        mode (GenerationMode): The mode of generation, either GENERATE with examples or \
            GENERATE_FT without examples on fine-tuned model or REFINE with examples.
        reasoning_effort (str): The reasoning effort level for models that support it (default is "medium").

    Returns:
        tuple[str, list[dict], dict | None]: Response string, API context, and token usage dict (or None).
    """
    logger.info("Sending generation request to OpenAI API...")
    logger.debug(f"System prompt: {system_prompt_str}")
    for user_prompt_str in user_prompt_str_list:
        logger.debug(f"User prompt: {pprint.pformat(user_prompt_str)}")
    logger.debug(f"User request: {user_request_str}")
    logger.debug(f"Model: {model}")
    if image_paths:
        logger.debug(f"Image paths ({len(image_paths)}): {image_paths}")

    messages = [{"role": "developer", "content": [{"type": "input_text", "text": system_prompt_str}]}]

    for user_prompt_str in user_prompt_str_list:
        usr_prompt = {"role": "user", "content": [{"type": "input_text", "text": user_prompt_str}]}
        messages.append(usr_prompt)

    if image_paths:
        image_content: list[dict[str, str]] = [{"type": "input_text", "text": RIG_RENDER_USER_PROMPT}]
        for path in image_paths:
            image_content.append({"type": "input_image", "image_url": _encode_image_base64(path)})
        messages.append({"role": "user", "content": image_content})

    usr_request = {"role": "user", "content": [{"type": "input_text", "text": user_request_str}]}

    messages.append(usr_request)

    client = OpenAI()

    args: dict[str, Any] = {
        "model": model,
        "input": messages,
        "text": {"format": {"type": "json_object"} if mode != GenerationMode.GENERATE_FT else {"type": "text"}},
        "service_tier": _system_config.service_tier.value,
    }

    if use_code_interpreter:
        args["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        args["include"] = ["code_interpreter_call.outputs"]

    o_series_regex = r"^o[134]$"
    if (model.startswith("gpt-5") and "chat" not in model) or re.match(o_series_regex, model):
        args["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
    else:
        args["temperature"] = temperature
        args["top_p"] = top_p

    logger.debug(f"API input messages: {pprint.pformat(_sanitize_messages_for_log(messages))}")

    response = client.responses.create(**args)  # type: ignore

    response_str = response.output_text
    token_usage = response_usage_to_dict(getattr(response, "usage", None))

    logger.debug(f"Tokens usage: {response.usage}")
    logger.debug(f"Output: {response.output}")

    if response_str is None or response_str.strip() == "":
        raise ValueError("Response from OpenAI API is empty or None.")

    return response_str, messages, token_usage
