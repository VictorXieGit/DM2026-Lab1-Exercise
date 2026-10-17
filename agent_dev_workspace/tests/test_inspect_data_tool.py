import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_inspect_data_tool():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load student's data tools
    student_tools = make_tools(session)
    inspect_data_tool = next(t for t in student_tools if t.name == "inspect_data_tool")
    
    # Load the fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Test inspecting default number of rows (5)
    result_default = inspect_data_tool.invoke({})
    assert result_default["n_rows_inspected"] == 5
    assert result_default["total_rows"] == 8
    assert "text" in result_default["columns"]
    assert len(result_default["preview"]) == 5
    
    # Test inspecting a custom number of rows (3)
    result_custom = inspect_data_tool.invoke({"n": 3})
    assert result_custom["n_rows_inspected"] == 3
    assert len(result_custom["preview"]) == 3
