from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal

from common import demo_image_url as image_url
from common import hf_mm_model as model_name
from common.images import show_demo_image
from common.models import default_model_kwargs, demo_multi_modal

model_kwargs = default_model_kwargs()

show_demo_image()
mm_model = HuggingFaceMultiModal.from_model_name(
    model_name, trust_remote_code=True, **model_kwargs
)
demo_multi_modal(
    mm_model, model_name, [ImageNode(image_path=image_url)], streaming=False
)
