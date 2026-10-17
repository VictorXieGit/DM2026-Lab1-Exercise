import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool():
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load student's data tools
    student_tools = make_tools(session)
    tokenize_tool = next(t for t in student_tools if t.name == "tokenize_tool")
    
    # Load the fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Run tokenize_tool
    result = tokenize_tool.invoke({})
    
    # Check return structure and document count
    assert result["n_documents"] == 8
    
    # Expected tokens per TEST_FIXTURE.md #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []  # empty text row
    ]
    
    assert session.artifacts["tokens"] == expected_tokens
    assert session.dataframe["unigrams"].tolist() == expected_tokens
