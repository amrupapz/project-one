import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PromptConfig
from tests.conftest import VALID_PROMPT

REPO_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_loads_valid_prompt(write_prompt):
    config = PromptConfig.from_yaml(write_prompt(VALID_PROMPT))
    assert config.version == "v1"
    assert config.model_name == "openai/gpt-4o-mini"
    assert config.temperature == 0
    assert config.few_shot_examples[0].output.category == "billing"


@pytest.mark.parametrize("name", ["v1", "v2"])
def test_repo_prompts_are_valid(name):
    config = PromptConfig.from_yaml(REPO_PROMPTS / f"{name}.yaml")
    assert config.version == name


def test_temperature_defaults_to_zero(write_prompt):
    data = copy.deepcopy(VALID_PROMPT)
    del data["temperature"]
    assert PromptConfig.from_yaml(write_prompt(data)).temperature == 0.0


def test_few_shot_examples_optional(write_prompt):
    data = copy.deepcopy(VALID_PROMPT)
    del data["few_shot_examples"]
    assert PromptConfig.from_yaml(write_prompt(data)).few_shot_examples == []


@pytest.mark.parametrize("field", ["version", "created_at", "model_name", "system_prompt"])
def test_missing_required_field_fails(write_prompt, field):
    data = copy.deepcopy(VALID_PROMPT)
    del data[field]
    with pytest.raises(ValidationError, match=field):
        PromptConfig.from_yaml(write_prompt(data, name="v1"))


@pytest.mark.parametrize("bad_version", ["1", "V1", "v0", "v1.0", "version1", "v", ""])
def test_bad_version_id_fails(write_prompt, bad_version):
    data = {**VALID_PROMPT, "version": bad_version}
    with pytest.raises(ValidationError, match="version"):
        PromptConfig.from_yaml(write_prompt(data, name="whatever"))


def test_version_must_match_filename(write_prompt):
    with pytest.raises(ValueError, match="does not match file name"):
        PromptConfig.from_yaml(write_prompt(VALID_PROMPT, name="v2"))


@pytest.mark.parametrize("temperature", [-0.1, 2.1, 5])
def test_temperature_out_of_range_fails(write_prompt, temperature):
    data = {**VALID_PROMPT, "temperature": temperature}
    with pytest.raises(ValidationError, match="temperature"):
        PromptConfig.from_yaml(write_prompt(data))


@pytest.mark.parametrize("temperature", [0, 1.0, 2])
def test_temperature_bounds_inclusive(write_prompt, temperature):
    data = {**VALID_PROMPT, "temperature": temperature}
    assert PromptConfig.from_yaml(write_prompt(data)).temperature == temperature


@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "", "openai/"])
def test_bad_model_name_fails(write_prompt, model_name):
    data = {**VALID_PROMPT, "model_name": model_name}
    with pytest.raises(ValidationError, match="model_name"):
        PromptConfig.from_yaml(write_prompt(data))


def test_unknown_field_fails(write_prompt):
    data = {**VALID_PROMPT, "temprature": 0.5}
    with pytest.raises(ValidationError, match="temprature"):
        PromptConfig.from_yaml(write_prompt(data))


def test_few_shot_output_is_validated(write_prompt):
    data = copy.deepcopy(VALID_PROMPT)
    data["few_shot_examples"][0]["output"]["category"] = "shipping"
    with pytest.raises(ValidationError, match="category"):
        PromptConfig.from_yaml(write_prompt(data))


def test_non_mapping_yaml_fails(tmp_path):
    path = tmp_path / "v1.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        PromptConfig.from_yaml(path)
