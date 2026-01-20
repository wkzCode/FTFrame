from __future__ import annotations

from typing import List, Optional


def build_mcq_prompt_simple_exam(subject: Optional[str], question: str, choices: List[str]) -> str:
    """Prompt template closely matching your current script."""
    subject_line = f"Subject: {subject}\n" if subject else ""
    lines = [
        "You are taking a multiple-choice exam.",
        "Choose the correct answer.",
        "Respond with ONLY the letter: A, B, C, or D.",
        "",
        subject_line + f"Question: {question}",
        "Choices:",
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
    ]
    return "\n".join([ln for ln in lines if ln != ""]) 


def build_prompt(template: str, subject: Optional[str], question: str, choices: List[str]) -> str:
    if template == "simple_exam":
        return build_mcq_prompt_simple_exam(subject, question, choices)
    raise ValueError(f"Unknown prompt template: {template}")
