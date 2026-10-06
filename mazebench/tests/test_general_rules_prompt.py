"""The method asks for general mechanisms instead of special cases and patches."""
from __future__ import annotations

from agent.world_model.tools import build_system_prompt


def test_method_asks_for_general_mechanisms():
    prompt = build_system_prompt(None)
    assert "5. Model GENERAL mechanisms." in prompt
    assert "Never\n   special-case particular rooms" in prompt
    assert "restructure the model substantially" in prompt


def test_general_rule_sits_between_method_and_memory():
    prompt = build_system_prompt(None)
    assert prompt.index("4. Guess boldly") < prompt.index("5. Model GENERAL") < prompt.index("- Keep DURABLE memory")
