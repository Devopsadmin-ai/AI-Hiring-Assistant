from typing import Literal, Union
from pydantic import BaseModel


class InterviewPlanRequest(BaseModel):
    job_id: int | None = None
    candidate_id: int | None = None
    question_min: int | None = None
    question_max: int | None = None


class FocusArea(BaseModel):
    area: str = ""
    priority: Literal["High", "Medium", "Low"] = "Medium"
    reason: str = ""


class StandardQuestionItem(BaseModel):
    id: int = 0
    question: str = ""
    focus_area: str = ""
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    hints: str = ""
    possible_answers: list[str] = []


class CodingQuestionItem(BaseModel):
    id: int = 0
    title: str = ""
    task: str = ""
    focus_area: str = ""
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    example: str = ""
    input: str = ""
    output: str = ""


class QuestionCategory(BaseModel):
    category: Literal["Technical", "Behavioural", "Logical", "Coding"] = "Technical"
    questions: list[Union[StandardQuestionItem, CodingQuestionItem]] = []


class InterviewPlan(BaseModel):
    interview_focus_areas: list[FocusArea] = []
    question_categories: list[QuestionCategory] = []


class InterviewPlanResponse(BaseModel):
    interview_plan: InterviewPlan = InterviewPlan()
