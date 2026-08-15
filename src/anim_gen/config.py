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

from dataclasses import dataclass, fields
from enum import Enum

from dotenv import load_dotenv

# TODO: Add a flag for the humanoid generation, which will change a prompt template
# and consider only rotational transformations (and maybe translations for root joint)
load_dotenv(".env.keys", override=False)


class GenerationMode(Enum):
    GENERATE = "generate"
    GENERATE_FT = "generate_ft"
    REFINE = "refine"


class OpenAIServiceTier(Enum):
    AUTO = "auto"
    DEFAULT = "default"
    FLEX = "flex"
    SCALE = "scale"
    PRIORITY = "priority"


@dataclass
class SystemConfig:
    # Logging
    log_level: str = "DEBUG"
    log_dir: str = "logs"

    # Retries
    num_retries: int = 2

    # Interpolation
    interpolation_type: str = "auto"
    quat_interpolation_type: str = "slerp"

    # Model defaults
    gen_model: str = "gpt-5.4"
    refine_model: str = "gpt-5.4"
    selection_model: str = "gpt-5.4"
    motion_description_model: str = "gpt-5.4"
    cleanup_model: str = "gpt-5.4"
    prompt_validation_model: str = "gpt-4o"
    gen_temperature: float = 1.0
    refine_temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    service_tier: OpenAIServiceTier = OpenAIServiceTier.DEFAULT

    # Keyframe sampling
    sq_sampling_error: float = 0.01
    sampling_cluster_tolerance: int = 2
    decouple_transformations: bool = True

    # Transformations
    use_euler_angles: bool = True
    bind_relative_transformations: bool = True
    world_bind_transformations: bool = False

    # Object JSON bind pose components
    include_bind_translations: bool = True
    include_bind_rotations: bool = True
    include_bind_scales: bool = True

    # Rig rendering
    view_angle_offset: float = 20.0
    render_rig_bones: bool = True
    hide_view_normal_gizmo_axis: bool = True

    # Tools
    use_code_interpreter: bool = True

    # Validation sets
    _valid_log_levels: set[str] | None = None
    _valid_interpolation_types: list[str] | None = None
    _valid_quat_interpolation_types: list[str] | None = None

    def __post_init__(self):
        if self._valid_log_levels is None:
            self._valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self._valid_interpolation_types is None:
            self._valid_interpolation_types = ["auto", "linear"]
        if self._valid_quat_interpolation_types is None:
            self._valid_quat_interpolation_types = ["slerp", "smooth_slerp", "bezier_slerp"]

    def to_dict(self) -> dict:
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, set):
                value = list(value)
            if isinstance(value, Enum):
                value = value.value
            if field.name.startswith("_"):
                continue
            result[field.name] = value
        return result


_system_config = SystemConfig()


def get_system_config() -> SystemConfig:
    return _system_config


def set_system_config(config: SystemConfig) -> None:
    global _system_config
    _system_config = config


# ===== PROMPT TEMPLATES =====
SELECTION_SYS_PROMPT_STR_RAW = r"""# Task
You are a highly-trained professional in semantic analysis and selection of 3D animations examples.
You are provided with a metadata of 20 distinct 3D animations of the same 3D model below (# Metadata for selection).
The animations differ in their motion, skeleton/rig type, transformation types they contain, semantic and emotional meaning, context, duration, etc.

These animations serve as a few-shot examples for the LLM model responsible for generating new 3D animations based on the user's request.
The user will provide you with a natural language description of the animation they want to generate.
Your task is to select the most relevant example animations from the provided list of examples, adhering to the selection strategy below (# Selection strategy).

The output must be strictly following the output format below (# Output format).

# Examples metadata format
The metadata is a JSON object, where each entry is a key-value pair, where the key is the name of the animation, and the value is the metadata of the animation.

Example entry:
```json
"<animation_name>":{
  "caption":{
    "motion_description":"<motion_description>",
    "semantic_tags":["<semantic_tag_1>","<semantic_tag_2>"],
    "contextual_hints":["<contextual_hint_1>","<contextual_hint_2>"]
  },
  "rig_type":"<rig_type>",
  "additional_joints":["<additional_joint_1>","<additional_joint_2>"],
  "duration":<duration>,
  "total_keyframes":<total_keyframes>,
  "transformation_ratios":{"translation":<translation_ratio>,"rotation":<rotation_ratio>,"scale":<scale_ratio>}
}
```

- `caption` is a dictionary containing the textual description of the animation:
    - `motion_description` is a detailed natural language description of the motion of the animation. It is the most important field in the caption and serves as the main criterion for the selection.
    - `semantic_tags` is a list of tags that suggest the emotional and expressive meaning of the animation.
    - `contextual_hints` is a list of tags that suggest the context in which the animation is used.
- `rig_type` is the type of the rig used in the animation. Possible values are "simple" and "complex".
- `additional_joints` is a list of additional joints that are used in the animation.
- `duration` is the duration of the animation in seconds.
- `total_keyframes` is the total number of keyframes (key moments) in the animation. Higher number of keyframes indicates more complex motion.
- `transformation_ratios` is a dictionary containing the ratios of the transformations in the animation. The ratios reflect how much of the animation is based on translations, rotations, and scales.

# Selection strategy:
You must select the example animations strictly adhering to the below rules:

- You must select at least one but not more than three example animations. You should always select the most relevant example animations based on the user's request. Adjust the number of selected examples based on how closely the example animations match the user's request (more matches, more examples).
- The order of the selected examples is important. Most relevant examples should be placed at the beginning of the list.
- The first example you select serves as the base for the generation. It is the most important example. All following examples must have the same or simpler rig type as the first example, i.e. if the first example's rig type is "simple", all following examples must have "simple" rig type; if the first example's rig type is "complex", the following examples may have "simple" rig type, as well as "complex" rig type.
- The main criterion for the selection is the similarity of the motion description of the example animation and the user's request. You should find the most relevant examples based on their semantic meaning.
- The secondary criterion that have a lower priority is the transformation ratios of the example animation. You should imagine how the user requested animation would look like in terms of translations, rotations, and scales. Include examples that are more representative of the transformation types you think will be necessary to fulfill the user's request. This criterion also serves as a fallback for the main criterion, when no examples are similar enough to the user's request, select based on this criterion.
- The third criterion that have the lowest priority is semantic tags and contextual hints. They serve as a hint for the selection, and can be used to help you sort the examples by their relevance.
- Other fields may serve as an additional hint for the selection, as they may help to build a more complete picture of the example animation.

# Output format:
The output must be a JSON object with the following fields:
- `selected_examples`: a list of strings, where each string is the name of the selected example animation.
- `reasoning`: a string containing the reasoning process of the selection.
- You are not allowed to include any additional animations in the output that are not present in the provided metadata.

Example output:
```json
{
  "selected_examples": ["<animation_name_1>", "<animation_name_2>", "<animation_name_3>"],
  "reasoning": "<reasoning>"
}
```

# Metadata for selection
{EXAMPLES_METADATA}
"""

# ===== MOTION DESCRIPTION OPTIMIZATION PROMPTS =====
MOTION_DESCRIPTION_GENERATE_SYS_PROMPT_STR_RAW = r"""# Task
You are a professional animator and motion description writer. Your task is to transform user prompts into clear, concise motion descriptions for 3D animations.

# Instructions
- Remove all irrelevant parts that do not form motion description (e.g., "Please generate", "I want", "Make an animation where", etc.)
- Fix any grammar errors
- Reformulate the prompt into a proper, natural motion description in present tense
- Keep the description concise but descriptive
- Focus on what the character/object does, not on the request itself

# Examples
- Input: "Please generate the animation where blob is putting on the headphones"
  Output: "The blob puts on the headphones"

- Input: "I want the character to wave hello"
  Output: "The character waves hello"

- Input: "Create an animation of a blob bouncing up and down"
  Output: "The blob bounces up and down"

# Output format
Output only the motion description text, nothing else. No explanations, no additional text, just the motion description.
"""

MOTION_DESCRIPTION_REFINE_SYS_PROMPT_STR_RAW = r"""# Task
You are a professional animator and motion description writer. Your task is to combine an existing motion description with a refinement request to create an updated motion description.

# Instructions
- Preserve the core motion from the original description
- Integrate the requested changes naturally into the description
- Use proper grammar and natural language flow
- Keep the description concise but descriptive
- If the refinement request asks to preserve the animation "as is" with additions, append the new actions after the original description

# Examples
- Original: "The blob puts on the headphones"
  Refinement: "Preserve the animation as is, except the blob should blink after putting on the headphones"
  Output: "The blob puts on the headphones, then blinks"

- Original: "The character waves hello"
  Refinement: "Make the wave slower and add a smile at the end"
  Output: "The character waves hello slowly, then smiles"

# Output format
Output only the updated motion description text, nothing else. No explanations, no additional text, just the motion description.
"""

PROMPT_VALIDATION_SYS_PROMPT_STR_RAW = r"""# Task
You are a validation system for a 3D animation generation tool. Your task is to determine if a user's animation request can be reliably generated with the tool's capabilities.

# Tool Capabilities
The tool can generate animations for:
- The "blob" character. It can also be called a "ball" or any named object that resembles a blob/sphere
- Animations using only the blob character and headphones accessory
- Any animations with headphones accessory are allowed
- Animations that use only skeletal animation (translations, rotations, and scales of joints)
- Simple deformations like squashing or stretching (achieved through joint scaling). Wobbling, bouncing, eye-blinking, etc., are allowed

# Tool Limitations
The tool CANNOT reliably generate animations that:
1. Request animation on objects different from the blob character (e.g., human, animal, car, robot, etc.)
2. Include usage of additional 3D assets except headphones (e.g., hourglass, cup of coffee, ball, table, etc.)
3. Require blendshape-based animation or morphing (e.g., blob morphs into cube shape, eyes become heart-shaped, eyes morph into wifi-signal, face morphing, shape-changing beyond simple squashing/stretching via joint scaling)

# Instructions
- Analyze the user's prompt carefully
- Check if it violates any of the three limitations above
- Respond with a JSON object containing:
  - "pass": boolean (true if valid, false if invalid)
  - "reason": string (optional, only include if pass is false) - brief explanation (1-2 sentences) of which limitation is violated and why
- Do not include any text outside the response JSON object.
- In the reason field, refer to the supported character as the "blob character". Do not mention validation in the reason, only provide the reason for the invalidity.
- Note that the user also can also provide refinement prompts. They are perfectly valid but should be validated under the same rules.
- Do not be too strict with the limitations and validation rules. It is always better to allow the request in case of doubt than to reject it.

# Examples
- Input: "The blob jumps up and down"
  Output: {"pass": true}

- Input: "The blob puts on headphones and does a flip"
  Output: {"pass": true}
  
- Input: "The blob throws the headphones away"
  Output: {"pass": true}

- Input: "A human character walks across the room"
  Output: {"pass": false, "reason": "The animation is requested on a human character, but the tool only supports the blob character."}

- Input: "The blob picks up a cup of coffee"
  Output: {"pass": false, "reason": "The animation includes usage of additional 3D assets (cup of coffee) that are not supported. Only headphones are allowed."}

- Input: "The blob's eyes morph into heart shapes"
  Output: {"pass": false, "reason": "The animation requires blendshape-based morphing (eyes changing shape), which is not supported. Only skeletal animations with simple deformations are allowed."}

- Input: "The blob squashes down when landing"
  Output: {"pass": true}
  
- Input: "Preserve the animation as is, except the blob should blink after putting on the headphones"
  Output: {"pass": true}
"""

# TODO: Add prompt variations for different modes of generation

SYS_PROMPT_STR_RAW = r"""# Task & identity:
You are an animator specialized in animating 3D rigged models. The model's skeleton is composed of several joints, each of which you can animate by modifying their positions, scales, and rotations over time. You will receive a JSON string (Object JSON), outlining the hierarchy of the model's skeleton and its bind pose, along with the textual description of the 3D model you need to animate.
You will also be given {NUM_EXAMPLES} examples of animations on this model, that include the natural language description of the animation, and JSON string (Animation JSON), specifying the transformations keyframes. You have to respond with a new animation that user requests.

# Input format:
## Object JSON format:
The "Object JSON" format defines the hierarchy of the model's skeleton and its bind pose. The "Object JSON" string is a dictionary, where:
- Each top-level key is the joint name, wherein it's value is the dictionary containing the {BIND_POSE_SPACE} bind pose transformations in the format: "p": [x, y, z], "r": [x, y, z], "s": [x, y, z]". Each of the components is optional and can be omitted in the Object JSON.
- 'p' is a list of three floats representing the x, y, z coordinates for position/translation; 'r' is a list of three floats representing the euler angles (x, y, z) for rotation/orientation (in degrees); and 's' is a list of three floats representing the scale in x, y, z directions.
- Joint names are hierarhical, e.g., 'root_j/body_main_j' notation means that the 'body_main_j' joint is a child of the 'root_j' joint. The root joint is the first joint in the name, and all other joints are its children. The hierarchy is defined by the order of the joints in the joint name.

## Animation JSON format:
The "Animation JSON" format defines the keyframes for each animated joint. The "Animation JSON" string is a dictionary, where:
- Each top-level key is the joint name, wherein it's value is the dictionary containing the keyframes that specify the {ANIM_TRANSFORM_TYPE} transformations in the format: "t: {'p': [x, y, z], 'r': [x, y, z], 's': [x, y, z]}".
- Each keyframe has a float key 't', representing the time stamp; the value optionally specifies (depending on animation) 'p', 'r', and 's' transformations.
    - Example: "'root_j/body_main_j': {0.00: {'p': [0.25, 0.17, -2.54], 'r': [90.0, 0.0, 0.0], 's': [1.00, 1.00, 1.00]}, ...}" means that at time 0.00, the joint 'body_main_j' (child of 'root_j') is at position [0.25, 0.17, -2.54], has rotation of 90 degrees around X-axis, 0 degrees around Y-axis, and 0 degrees around Z-axis, and scale [1.00, 1.00, 1.00].
- The joints that are not animated in the animation are not included in the "Animation JSON".
{RENDER_INPUT_FORMAT}
# Instructions:
- For the rotations, the keyframes should be set with a difference of less than 180 degrees between them, as we must know in which direction the joint is rotating.
- You must be sure to keyframe both the start and end of any atomic motion, i.e. "hold" a keyframe before any change. This allows us to establish a reference position and prevent unbounded interpolation by knowing when the motion starts and ends.
{RENDER_WORKFLOW}
## Important notes:
- The transformations are given in a Z-up right-handed coordinate system. Rotations follow the right-hand rule: a positive angle is counter-clockwise when looking from the positive end of the axis toward the origin.
- The transformations of each joint are in joint-local space, i.e., the transformations are relative to the parent joint. Therefore, take into account that joint's transformations affect the transformations of its children downwards in the hierarchy. Be extremely careful and think deeply about the hierarchical relationships when animating the joints, especially orientations.
{BIND_RELATIVE_NOTE}- The keyframes represent the key moments of the animation, where significant changes in transformations occur. You are required to generate keyframes based on the same strategy, i.e., at key moments. Note that the time steps 't' are not necessarily evenly spaced in time, but rather based on the actual motion of the joints.

# Output format:
The format of the output JSON must be the same as the Animation JSON format provided in the examples. You are not allowed to include any additional information or comments in the output JSON. The output JSON must be a valid JSON string.
"""

# - You must be sure to add the keyframe before the start of any motion of the joint in the animation, i.e. "hold" a keyframe before any change. This allows to establish a reference position for correct interpolation by specifying when exactly does the motion start and end.


USR_PROMPT_OBJ_STR_RAW = r"""The object you will animate is a {OBJECT_DESCRIPTION}

Object JSON:
```
{OBJECT_JSON}
```
"""

USR_PROMPT_ANIM_STR_RAW = r"""The {EXAMPLE_NUM} example of animation:
Animation description: "{ANIMATION_DESCRIPTION}"

Animation JSON:
```
{ANIMATION_STRING}
```
"""

USR_REQUEST_STR_RAW = r"""Generate a new animation:

Animation description: {REQ_ANIM_DESC}"""

USR_REQUEST_ADDENDUM_STR_RAW = (
    "Reason deeply and be rigorous. The fidelity and quality of the resulting "
    "animation should be extremely high. Ensure smoothness, completeness, "
    "appropriate duration and speed of the animation."
)


# ===== REFINE PROMPT TEMPLATES =====
SYS_PROMPT_REFINE_STR_RAW = r"""# Task & identity:
You are an animator specialized in refining, improving, and extending animations of 3D rigged models. The model's skeleton is composed of several joints, each of which you can animate by modifying their positions, scales, and rotations over time. You will receive a JSON string (Object JSON), outlining the hierarchy of the model's skeleton and its bind pose, along with its textual description.
You will also be given with the natural language description of the original animation, and JSON string (Animation JSON), specifying the transformations keyframes. Additionally, you may also be given some additional examples of animations on this model if user considers it necessary. You have to respond with a modified animation that fulfills user requests.

# Input format:
## Object JSON format:
The "Object JSON" format defines the hierarchy of the model's skeleton and its bind pose. The "Object JSON" string is a dictionary, where:
- Each top-level key is the joint name, wherein it's value is the dictionary containing the {BIND_POSE_SPACE} bind pose transformations in the format: "p": [x, y, z], "r": [x, y, z], "s": [x, y, z]".
- 'p' is a list of three floats representing the x, y, z coordinates for position/translation; 'r' is a list of three floats representing the euler angles (x, y, z) for rotation/orientation (in degrees); and 's' is a list of three floats representing the scale in x, y, z directions.
- Joint names are hierarhical, e.g., 'root_j/body_main_j' notation means that the 'body_main_j' joint is a child of the 'root_j' joint. The root joint is the first joint in the name, and all other joints are its children. The hierarchy is defined by the order of the joints in the joint name.

## Animation JSON format:
The "Animation JSON" format defines the keyframes for each animated joint. The "Animation JSON" string is a dictionary, where:
- Each top-level key is the joint name, wherein it's value is the dictionary containing the keyframes that specify the {ANIM_TRANSFORM_TYPE} transformations in the format: "t: {'p': [x, y, z], 'r': [x, y, z], 's': [x, y, z]}".  Each of the components is optional and can be omitted in the Object JSON.
- Each keyframe has a float key 't', representing the time stamp; the value optionally specifies (depending on animation) 'p', 'r', and 's' transformations.
    - Example: "'root_j/body_main_j': {0.00: {'p': [0.25, 0.17, -2.54], 'r': [90.0, 0.0, 0.0], 's': [1.00, 1.00, 1.00]}, ...}" means that at time 0.00, the joint 'body_main_j' (child of 'root_j') is at position [0.25, 0.17, -2.54], has rotation of 90 degrees around X-axis, 0 degrees around Y-axis, and 0 degrees around Z-axis, and scale [1.00, 1.00, 1.00].
- The joints that are not animated in the animation are not included in the "Animation JSON".
{RENDER_INPUT_FORMAT}
# Instructions:
- For the rotations, the keyframes should be set with a difference of less than 180 degrees between them, as we must know in which direction the joint is rotating.
- You must be sure to keyframe both the start and end of any atomic motion, i.e. "hold" a keyframe before any change. This allows us to establish a reference position and prevent unbounded interpolation by knowing when the motion starts and ends.
{RENDER_WORKFLOW}
## Important notes:
- The transformations are given in a Z-up right-handed coordinate system. Rotations follow the right-hand rule: a positive angle is counter-clockwise when looking from the positive end of the axis toward the origin.
- The transformations of each joint are in joint-local space, i.e., the transformations are relative to the parent joint. Therefore, take into account that joint's transformations affect the transformations of its children downwards in the hierarchy. Be extremely careful and think deeply about the hierarchical relationships when animating the joints, especially orientations.
{BIND_RELATIVE_NOTE}- The keyframes represent the key moments of the animation, where significant changes in transformations occur. You are required to generate keyframes based on the same strategy, i.e., at key moments. Note that the time steps 't' are not necessarily evenly spaced in time, but rather based on the actual motion of the joints.

# Output format:
The format of the output JSON must be the same as the Animation JSON format provided in the input. You are not allowed to include any additional information or comments in the output JSON. The output JSON must be a valid JSON string.
"""

USR_PROMPT_REFINE_OBJ_STR_RAW = r"""The object whose animation you will refine is a {OBJECT_DESCRIPTION}.

Object JSON:
```
{OBJECT_JSON}
```
"""

USR_PROMPT_REFINE_ANIM_STR_RAW = r"""The animation you need to refine:
Animation description: "{ANIMATION_DESCRIPTION}"

Animation JSON:
```
{ANIMATION_STRING}
```
"""

USR_PROMPT_REFINE_EXAMPLES_STR_RAW = r"""The {EXAMPLE_NUM} additional example of animation on this model:
Animation description: "{ANIMATION_DESCRIPTION}"

Animation JSON:
```
{ANIMATION_STRING}
```
"""

USR_REQUEST_REFINE_STR_RAW = r"""Modify the animation based on the following request:
Animation modification description: {REQ_ANIM_DESC}"""


# ===== JOINT NAME CLEANUP PROMPT =====
JOINT_CLEANUP_SYS_PROMPT_STR_RAW = r"""# Task
You are a specialist in 3D skeleton rigging conventions. You will receive a JSON list of joint/bone names from a 3D model's skeleton. Your task is to produce the **shortest, clearest** version of each name that **preserves its anatomical, structural, or semantic meaning**.

# Rules
1. Strip prefixes and suffixes that are repeated across most or all joints and carry no anatomical information (e.g. `mixamorig_`, `Bip01_`, `model_`, `_M`, `_J`, `_jnt`).
2. Keep laterality markers (`_L`, `_R`, `Left`, `Right`, `_l`, `_r`) — these are meaningful.
3. Keep trailing digits that distinguish segments of the same bone chain (e.g. `Spine1`, `Spine2`, `Tail0`, `Tail1`).
4. Keep `_end` or `_End` suffixes — they mark terminal joints.
5. Preserve the original casing style (PascalCase, camelCase, snake_case) of the meaningful part.
6. Every output name must be **non-empty** and **unique** — no two input names may map to the same output.
7. The mapping must be **complete** — every input name must appear as a key exactly once.
8. If the names are already concise and clean, return them unchanged (identity mapping).

# Examples

Input: `["mixamorig_Hips", "mixamorig_Spine", "mixamorig_Spine1", "mixamorig_LeftShoulder", "mixamorig_LeftArm", "mixamorig_Head", "mixamorig_HeadTop_End"]`
Output:
```json
{"mapping":{"mixamorig_Hips":"Hips","mixamorig_Spine":"Spine","mixamorig_Spine1":"Spine1","mixamorig_LeftShoulder":"LeftShoulder","mixamorig_LeftArm":"LeftArm","mixamorig_Head":"Head","mixamorig_HeadTop_End":"HeadTop_End"}}
```

Input: `["model_Root_M", "model_Tail0_M", "model_Hip_R", "model_Hip_L", "model_Head_M"]`
Output:
```json
{"mapping":{"model_Root_M":"Root","model_Tail0_M":"Tail0","model_Hip_R":"Hip_R","model_Hip_L":"Hip_L","model_Head_M":"Head"}}
```

Input: `["body", "head", "antenna01_R", "antenna02_R"]`
Output:
```json
{"mapping":{"body":"body","head":"head","antenna01_R":"antenna01_R","antenna02_R":"antenna02_R"}}
```

# Output format
Return a single JSON object with one key `"mapping"` whose value is an object mapping every input name to its cleaned name. No other keys, text, or explanation."""


# ===== RENDER IMAGE PROMPT FILLS =====
# Used to conditionally populate {RENDER_INPUT_FORMAT} and {RENDER_WORKFLOW} in the system prompts
# when rig render images are provided as input to the generation model.

RENDER_INPUT_FORMAT_FILL = (
    "\n## Model render images\n"
    "You will also be provided with rendered images of the 3D model from multiple viewpoints. "
    "Each image shows the model's mesh with a skeleton (rig) overlay, where joints are "
    "visualized on top of the mesh. The images also include a color-coded axis gizmo "
    "(X axis = red, Y axis = green, Z axis = blue) that indicates the spatial orientation of the model.\n"
    "Note that the bones (blue cylinders between red joints) are purely for visualization purposes and are not "
    "part of the model's skeleton. The animation is authored on joints only, not on bones in any way."
)

RENDER_WORKFLOW_FILL = (
    "- First, carefully examine the provided render images of the model. Assess the model's appearance, "
    "proportions, and overall shape. Identify where each joint is positioned on the model and how they "
    "connect. Pay close attention to the axis gizmo overlay to understand spatial orientation: which "
    "direction is left/right, forward/backward, and up/down relative to the model.\n"
    "- Then, study the skeletal structure and bind pose from the Object JSON. Cross-reference the joint "
    "hierarchy with what you observe in the renders to build a complete understanding of how the model "
    "is rigged.\n"
    "- Carefully and rigorously examine the provided animation examples to understand the conventions, "
    "typical keyframe patterns, and motion styles used for this model.\n"
    "- Finally, plan the animation before generating keyframes. Deeply reason about the motion, which "
    "joints are involved, how and when they should move, and how they relate to each other "
    "hierarchically. Based on your analysis, carefully craft the keyframes."
)

DEFAULT_WORKFLOW_FILL = (
    "- Always plan the animation in advance before generating the keyframes. Deeply reason about "
    "the motion, which joints are involved in it, how and when should they move, how are they "
    "related to each other, etc. Based on your reasoning, carefully craft the keyframes."
)

RIG_RENDER_USER_PROMPT = (
    "Rendered views of the 3D model from multiple viewpoints, showing the mesh "
    "with skeleton (rig) overlay and color-coded axis gizmo (X=red, Y=green, Z=blue):"
)


def get_prompt_strings(mode: GenerationMode = GenerationMode.GENERATE) -> dict:
    """
    Returns the prompt strings for the animation generation process based on the specified mode.

    Parameters:
        mode (GenerationMode): The mode of generation.
        config (Optional[Config]): The configuration object containing various settings (in particular prompt templates).

    Returns:
        dict: A dictionary containing the system prompt string, user prompt object string, user request string, and user prompt animation string (if applicable).
    """
    match mode:
        case GenerationMode.GENERATE:
            return {
                "system": SYS_PROMPT_STR_RAW,
                "user_object": USR_PROMPT_OBJ_STR_RAW,
                "user_request": USR_REQUEST_STR_RAW,
                "user_request_addendum": USR_REQUEST_ADDENDUM_STR_RAW,
                "user_animation": USR_PROMPT_ANIM_STR_RAW,
            }
        case GenerationMode.REFINE:
            return {
                "system": SYS_PROMPT_REFINE_STR_RAW,
                "user_object": USR_PROMPT_REFINE_OBJ_STR_RAW,
                "user_request": USR_REQUEST_REFINE_STR_RAW,
                "user_request_addendum": USR_REQUEST_ADDENDUM_STR_RAW,
                "user_animation": USR_PROMPT_REFINE_ANIM_STR_RAW,
                "user_examples": USR_PROMPT_REFINE_EXAMPLES_STR_RAW,
            }
        case _:
            raise ValueError(f"Unsupported generation mode: {mode}. Supported modes are: {list(GenerationMode)}")
