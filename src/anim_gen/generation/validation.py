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
import os

from openai import OpenAI, OpenAIError

from ..config import GenerationMode, get_system_config
from ..data_structs import Config

logger = logging.getLogger(__name__)

_system_config = get_system_config()


def _validate_inputs(input_files: list[dict], mode: GenerationMode) -> None:
    """
    Validate the inputs for the animation generation process.
    """
    if len(input_files) == 0:
        logger.error("No input files provided.")
        raise ValueError("No input files provided.")

    if len(input_files) > 4:
        logger.warning(
            "Providing more than 4 input files may cause context degradation and cause bad results. "
            "Consider reducing the number of input files."
        )

    if mode == GenerationMode.GENERATE_FT:
        logger.error("Fine-tuned generation is not implemented yet.")
        raise NotImplementedError("Fine-tuned generation is not implemented yet.")

    # if mode == GenerationMode.REFINE and len(input_files) > 1:
    #     logger.error(f"Refinement requires exactly one input file.")
    #     raise ValueError("Refinement requires exactly one input file.")

    for input_file in input_files:
        if (not input_file["path"].lower().endswith(".usda")) and (not input_file["path"].lower().endswith(".usdc")):
            logger.error(f"File {input_file['path']} is not a USD file.")
            raise ValueError(f"File {input_file['path']} is not a USD file.")
        if not isinstance(input_file["caption"], dict):
            logger.error(f"Caption for {input_file['path']} must be a dictionary.")
            raise ValueError(f"Caption for {input_file['path']} must be a dictionary.")

        required_keys = ["model_description", "motion_description"]
        for key in required_keys:
            if key not in input_file["caption"]:
                logger.error(f"Caption for {input_file['path']} is missing required key: {key}")
                raise ValueError(f"Caption for {input_file['path']} is missing required key: {key}")

        for key in input_file["caption"].keys():
            if key not in required_keys:
                logger.warning(f"Caption for {input_file['path']} has unknown key '{key}' and will be ignored.")


def _validate_request(request: dict) -> None:
    if "description" not in request:
        logger.error("Request is missing required key: description")
        raise ValueError("Request is missing required key: description")

    for key in request.keys():
        if key not in ["description"]:
            logger.warning(f"Request has unknown key '{key}' and will be ignored.")


def _validate_config(config: Config) -> None:
    """
    Validate the configuration.
    """
    if config.interpolation_type is not None:
        if not isinstance(config.interpolation_type, str):
            raise ValueError("interpolation_type must be a string")
        valid_types = _system_config._valid_interpolation_types
        if valid_types is not None and config.interpolation_type not in valid_types:
            raise ValueError(f"Invalid interpolation type: {config.interpolation_type}")

    if config.model is not None:
        if not isinstance(config.model, str):
            raise ValueError("model must be a string")
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            models = client.models.list().data
            available_ids = {m.id for m in models}
        except OpenAIError as e:
            raise ValueError(f"Unable to fetch models via OpenAI API: {e}") from e
        if config.model not in available_ids:
            raise ValueError(f"Model '{config.model}' is not available from OpenAI API")

    if config.temperature is not None:
        if not isinstance(config.temperature, (float, int)):
            raise ValueError("temperature must be a number")
        if not (0.0 <= config.temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")

    if config.top_p is not None:
        if not isinstance(config.top_p, (float, int)):
            raise ValueError("top_p must be a number")
        if not (0.0 <= config.top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")
