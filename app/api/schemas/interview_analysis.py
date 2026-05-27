from pydantic import BaseModel
from typing import Literal


class InterviewAnalyticalRequest(BaseModel):
    jd_id: int | None = None
    candidate_id: int | None = None
    transcript_s3_url: str | None = None


class InterviewAnalysisItem(BaseModel):
    id: int = 0
    question: str = ""
    category: Literal["Technical", "Behavioural", "Logical", "Coding"] = "Technical"
    status: Literal["correct", "partially_correct", "incorrect"] = "partially_correct"
    received_answer: str = ""
    answer_summary: str = ""
    possible_answers: list[str] = []


class EvaluationSummary(BaseModel):
    conceptual_understanding: float = 0.0
    problem_solving: float = 0.0
    depth_of_answers: float = 0.0
    communication_clarity: float = 0.0
    domain_knowledge: float = 0.0
    confidence_structure: float = 0.0


class CriterionScore(BaseModel):
    score: float = 0.0
    max: float = 0.0


class InterviewBreakdown(BaseModel):
    conceptual_understanding: CriterionScore = CriterionScore()
    problem_solving: CriterionScore = CriterionScore()
    depth_of_answers: CriterionScore = CriterionScore()
    communication_clarity: CriterionScore = CriterionScore()
    domain_knowledge: CriterionScore = CriterionScore()
    confidence_structure: CriterionScore = CriterionScore()


class InterviewResult(BaseModel):
    overall_interview_score: float = 0.0
    interview_weighted: float = 0.0
    interview_breakdown: InterviewBreakdown = InterviewBreakdown()
    overview: str = ""
    evaluation_summary: EvaluationSummary = EvaluationSummary()
    core_strengths: list[str] = []
    critical_gaps: list[str] = []
    follow_up_questions: list[str] = []


class InterviewAnalyticalResponse(BaseModel):
    total_questions: int = 0
    interview_analysis: list[InterviewAnalysisItem] = []
    interview_result: InterviewResult = InterviewResult()
