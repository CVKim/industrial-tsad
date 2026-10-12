from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """strict configs: unknown keys are errors, so a typo in a preset cannot silently fall back to a default"""
    model_config = ConfigDict(extra="forbid", frozen=True)
