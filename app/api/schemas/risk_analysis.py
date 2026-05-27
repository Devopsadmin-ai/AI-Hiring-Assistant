from pydantic import BaseModel
from typing import Literal


class RiskAnalyzeRequest(BaseModel):
    candidate_id: int | None = None
    linkedin_url: str | None = None


class RiskRatings(BaseModel):
    resume_consistency: float = 0.0
    skill_authenticity: float = 0.0
    profile_validation: float = 0.0
    communication_authenticity: float = 0.0


class CriterionScore(BaseModel):
    score: float = 0.0
    max: float = 0.0


class RiskBreakdown(BaseModel):
    resume_consistency: CriterionScore = CriterionScore()
    job_stability: CriterionScore = CriterionScore()
    skill_authenticity: CriterionScore = CriterionScore()
    ai_detection: CriterionScore = CriterionScore()
    employment_gaps: CriterionScore = CriterionScore()
    profile_validation: CriterionScore = CriterionScore()
    communication_authenticity: CriterionScore = CriterionScore()


class RiskLevelItem(BaseModel):
    title: str = ""
    desc: str = ""
    percent: float = 0.0


class RiskAnalysisAI(BaseModel):
    overview: str = ""
    ai_risk_narrative: str = ""
    core_strengths: list[str] = []
    critical_gaps: list[str] = []
    overall_risk_level: list[RiskLevelItem] = []


class RiskAnalyzeResponse(BaseModel):
    risk_score: float = 0.0
    safety_score: float = 0.0
    risk_flag: Literal["low", "medium", "high"] = "medium"
    risk_weighted: float = 0.0
    linkedin_extraction_condition: Literal["extracted", "not extracted"] = "not extracted"
    risk_ratings: RiskRatings = RiskRatings()
    risk_breakdown: RiskBreakdown = RiskBreakdown()
    risk_analysis_ai: RiskAnalysisAI = RiskAnalysisAI()
