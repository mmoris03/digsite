from digsite.llm import DEFAULT_MODEL, OllamaModel, create_language_model


def test_models_are_served_by_ollama() -> None:
    model = create_language_model("qwen2.5:3b", url="http://gpu-box:11434")

    assert isinstance(model, OllamaModel)
    assert model.name == "qwen2.5:3b"


def test_the_default_model_is_the_one_the_project_is_tried_with() -> None:
    assert create_language_model().name == DEFAULT_MODEL == "gemma3:4b"
