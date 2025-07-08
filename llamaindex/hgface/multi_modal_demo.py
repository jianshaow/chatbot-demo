from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal

from common import hf_mm_model as model_name
from common.images import show_demo_image
from common.models import default_model_kwargs, demo_multi_modal_legacy

model_kwargs = default_model_kwargs()

show_demo_image()
mm_model = HuggingFaceMultiModal.from_model_name(
    model_name, trust_remote_code=True, additional_kwargs=model_kwargs
)
demo_multi_modal_legacy(mm_model, model_name, streaming=False)
