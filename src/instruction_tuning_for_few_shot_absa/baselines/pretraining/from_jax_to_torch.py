from transformers import (
    FlaxT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import jax
from jax import numpy as jnp

tokenizer = T5Tokenizer.from_pretrained("t5-base-absa_3000/")
tokenizer.save_pretrained("t5-base-absa_pytorch")
model = T5ForConditionalGeneration.from_pretrained("t5-base-absa_3000/", from_flax=True)
model.save_pretrained("t5-base-absa_pytorch/")
