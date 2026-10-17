"""Tests for tools_data.py including load_dataset_tool and check_missing_tool."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool():
    """Test check_missing_tool against TEST_FIXTURE.md #2."""
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
    check_missing = next(t for t in student_tools if t.name == "check_missing_tool")

    result = check_missing.invoke({})

    assert result["missing_count"] == 1
    assert 7 in result["missing_indices"]
