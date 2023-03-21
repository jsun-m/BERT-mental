"""
Get started with our docs, https://docs.beam.cloud
"""
import beam
from constants import VOLUME_BASE_PATH
from beam.types import GpuType

app = beam.App(
    name="nlp-project-app",
    cpu=16,
    memory="24Gi",
    gpu=GpuType.A10G,
    python_packages="requirements.txt"
)

app.Mount.SharedVolume(
    name = "data",
    path = VOLUME_BASE_PATH,
)

app.Trigger.Webhook(
    inputs={
        "model_name": beam.Types.String(required=False),
        "dataset": beam.Types.String(required=False),
        "label_count": beam.Types.Float(required=False),
    },
    handler="run.py:train"
)
