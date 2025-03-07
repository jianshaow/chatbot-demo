import llama_index.multi_modal_llms.ollama.base
from llama_index.multi_modal_llms.ollama import OllamaMultiModal

from common import ollama_base_url as base_url
from common import ollama_mm_model as model
from common.images import show_demo_image
from common.models import demo_multi_modal
from common.ollama import get_additional_kwargs_from_model

llama_index.multi_modal_llms.ollama.base.get_additional_kwargs = (
    get_additional_kwargs_from_model
)

show_demo_image()
mm_model = OllamaMultiModal(base_url=base_url, model=model)
demo_multi_modal(mm_model, model)
