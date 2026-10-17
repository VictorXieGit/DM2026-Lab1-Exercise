"""Tests specifically for sample_data_tool in tools_data.py."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_sample_data_tool():
    """Test sample_data_tool with n=4 and random_state=42 against TEST_FIXTURE.md #13."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get student tools
    student_tools = make_tools(session)
    sample_data = next(t for t in student_tools if t.name == "sample_data_tool")

    result = sample_data.invoke({"n": 4, "random_state": 42})

    assert result["n_sampled"] == 4
    assert result["sample_indices"] == [1, 5, 0, 7]
