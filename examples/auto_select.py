import datetime
import os
import shutil

from anim_gen.config import GenerationMode
from anim_gen.data_structs import AnimationFile, Config
from anim_gen.generation.generation import generate_animation

with open("/path/to/openai_key.txt") as f:
    openai_key = f.read().strip()
    
os.environ["OPENAI_API_KEY"] = openai_key

prompt = "A blob suddenly performs a flip and lands on its back, then gets up back into the original position."
generation_request = {
  'description': prompt,
}

generation_config = Config(
    interpolation_type="auto",
    model="gpt-5.4",
    reasoning_effort="medium",
)

output_dir = "./gen_results"
animation_dir = f"{output_dir}/{datetime.datetime.now().strftime('%Y_%m_%d/%H_%M_%S')}"
os.makedirs(animation_dir, exist_ok=True)
animation_dir = os.path.abspath(animation_dir)

# Generate the animation
try:
    result = generate_animation(
        dest_path=f"{animation_dir}/result_generated.usda",
        request=generation_request,
        base_file=None,
        animation_examples=None,
        mode=GenerationMode.GENERATE,
        config=generation_config,
        logs_path=f"{animation_dir}/generation.log",
        metadata_path=f"{animation_dir}/metadata.json",
        render_video=True,
        render_rig=True,
        render_path=f"{animation_dir}/result_generated.mp4",
        auto_select_examples=True,
    )
except Exception:
    pass
