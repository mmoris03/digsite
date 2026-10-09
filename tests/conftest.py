import pytest


@pytest.fixture
def pizza_index():
    """The inverted index of five pizzas, written by hand.

    It is what `build_index` must make from their ingredients, and what the
    search and ranking tests start from, so they do not depend on
    `build_index`. A document's id is the position of its pizza in the list,
    and no pizza repeats an ingredient, so every count is 1.
    """
    return {
        "anchovies":    {3: 1},
        "artichokes":   {0: 1},
        "fontina":      {2: 1},
        "garlic":       {1: 1},
        "gorgonzola":   {2: 1},
        "mozzarella":   {0: 1, 2: 1, 3: 1, 4: 1},
        "mushrooms":    {0: 1},
        "oil":          {0: 1, 1: 1, 3: 1, 4: 1},
        "olives":       {0: 1},
        "oregano":      {1: 1, 3: 1, 4: 1},
        "sausage":      {4: 1},
        "stracchino":   {2: 1},
        "tomato":       {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
    }
