"""Immutable NuExtract3 model artifact manifest."""

MODEL_ID = "numind/NuExtract3"
MODEL_REVISION = "2e9fca82ee641e6bb6e1f5d905241e994be27a07"
BAKED_MODEL_PATH = "/opt/models/nuextract3"

MODEL_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model_mtp.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_LICENSE_FILE = "LICENSE"
MODEL_WEIGHT_FILES = ("model.safetensors", "model_mtp.safetensors")

# Values published by the Hugging Face API for the immutable revision above.
VERIFIED_LFS_FILES = (
    (
        "model.safetensors",
        9_078_620_504,
        "aca0a9d61da5df4fa4b1475b68c0a7205e5f8f5f20beb5055fde0622991f9ed7",
    ),
    (
        "model_mtp.safetensors",
        241_200_704,
        "7f993d7b896c6d3c72ee66fd446b28bcf316d5f5ce4a0427c0442dfe461cbe1b",
    ),
    (
        "tokenizer.json",
        19_989_343,
        "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    ),
)
MODEL_WEIGHT_BYTES = 9_319_821_208
