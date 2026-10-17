"""Tests specifically for check_duplicates_tool in tools_data.py."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_duplicates_tool():
    """Test check_duplicates_tool and dropping with keep=False against TEST_FIXTURE.md #3."""
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
    check_duplicates = next(t for t in student_tools if t.name == "check_duplicates_tool")

    # Check duplicates without dropping
    result_check = check_duplicates.invoke({"drop": False})
    assert result_check["n_duplicates"] == 2  # rows 5 and 6 are duplicates of each other
    assert 5 in result_check["duplicate_indices"]
    assert 6 in result_check["duplicate_indices"]
    assert len(session.dataframe) == 8

    # Now drop duplicates with keep=False (leaving 6 rows)
    result_drop = check_duplicates.invoke({"drop": True})
    assert result_drop["dropped"] is True
    assert len(session.dataframe) == 6
