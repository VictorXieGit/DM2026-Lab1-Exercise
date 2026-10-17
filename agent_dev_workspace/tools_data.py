"""Data inspection, validation, and preprocessing tools for the agentic pipeline."""

from typing import List, Optional, Dict, Any
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that instantiates and returns all data tools bound to the session."""

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """List files and directories in the workspace repo relative to subdirectory.
        
        Args:
            subdirectory: Relative path to list files from (default: '.').
        """
        import os
        
        try:
            full_path = os.path.normpath(subdirectory)
            if not os.path.exists(full_path):
                summary = {"status": "error", "error": f"Directory not found: {subdirectory}"}
                return summary
            
            entries = os.listdir(full_path)
            files = [f for f in entries if os.path.isfile(os.path.join(full_path, f))]
            directories = [d for d in entries if os.path.isdir(os.path.join(full_path, d))]
            
            summary = {
                "result_id": session.next_result_id("list_files"),
                "subdirectory": subdirectory,
                "files": sorted(files),
                "directories": sorted(directories)
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary
        except Exception as e:
            summary = {"status": "error", "error": str(e)}
            return summary

    @tool
    def tokenize_tool() -> dict:
        """Tokenize each document's text into unigrams using nltk word_tokenize after lowercasing.
        
        Stores token lists in session.dataframe['unigrams'] and session.artifacts['tokens'].
        """
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import word_tokenize

        if session.dataframe is None:
            return {"status": "error", "error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"status": "error", "error": "'text' column missing from dataframe."}

        # Tokenize text column: lowercase then word_tokenize (empty list for empty/NaN text)
        tokens_list = []
        for text in df["text"]:
            if isinstance(text, str) and text.strip():
                tokens = word_tokenize(text.lower())
            else:
                tokens = []
            tokens_list.append(tokens)

        df["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": len(tokens_list),
            "sample_tokens": tokens_list[:3] if tokens_list else []
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peek at the first n rows of the working dataframe.
        
        Args:
            n: Number of rows to inspect (default: 5).
        """
        if session.dataframe is None:
            return {"status": "error", "error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        head_df = df.head(n)

        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "n_rows_inspected": len(head_df),
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": head_df.to_dict(orient="records")
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Check for missing or empty text values in the working dataframe.
        
        Identifies rows where the 'text' column is null, NaN, or whitespace-only/empty.
        """
        if session.dataframe is None:
            return {"status": "error", "error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"status": "error", "error": "'text' column missing from dataframe."}

        # Find rows with missing or empty text
        missing_mask = df["text"].isna() | df["text"].apply(lambda x: not isinstance(x, str) or not x.strip())
        missing_indices = df.index[missing_mask].tolist()
        missing_count = len(missing_indices)

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "missing_count": missing_count,
            "missing_indices": missing_indices
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Check for duplicate documents in the working dataframe, and optionally drop them.
        
        Args:
            drop: If True, drop duplicate rows using keep=False (removing all copies of duplicates).
        """
        if session.dataframe is None:
            return {"status": "error", "error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        duplicates_mask = df.duplicated(keep=False)
        duplicate_indices = df.index[duplicates_mask].tolist()
        n_duplicates = len(duplicate_indices)

        if drop:
            df.drop_duplicates(keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "current_rows": len(df)
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> dict:
        """Subsample n rows randomly from the working dataframe.
        
        Args:
            n: Number of rows to sample (default: 5).
            random_state: Random seed for reproducibility (default: None).
        """
        if session.dataframe is None:
            return {"status": "error", "error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        sample_df = df.sample(n=min(n, len(df)), random_state=random_state)
        sample_indices = sample_df.index.tolist()

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "n_sampled": len(sample_df),
            "sample_indices": sample_indices,
            "sample_preview": sample_df.to_dict(orient="records")
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    return [list_files_tool, tokenize_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool]
