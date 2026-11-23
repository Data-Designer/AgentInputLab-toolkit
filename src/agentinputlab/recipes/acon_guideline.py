"""Public source-locked ACON guideline candidate producer.

Native analysis/update are bridged by workers.acon; generated templates remain
opaque until separately reviewed and validated for deployment.
"""

from workers.acon.fitted import ACONGuidelineArtifact, validate_fitted_template
from workers.acon.guideline import (
    ACONGuidelineControls,
    ACONGuidelinePipeline,
    validate_guideline_checkout,
    validate_training_bundle,
)

__all__ = [
    "ACONGuidelineArtifact",
    "ACONGuidelineControls",
    "ACONGuidelinePipeline",
    "validate_fitted_template",
    "validate_guideline_checkout",
    "validate_training_bundle",
]
