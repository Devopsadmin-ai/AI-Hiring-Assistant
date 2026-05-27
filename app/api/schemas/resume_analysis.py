from pydantic import BaseModel
from typing import Literal


class ResumeAnalyzeRequest(BaseModel):
    jd_id: int | None = None
    candidate_id: int | None = None
    resume_s3_url: str | None = None
    benchmark_s3_url: str | None = None


class SkillItem(BaseModel):
    name: str = ""
    category: Literal["technical", "soft", "domain"] = "technical"
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"] = "beginner"


class ExperienceItem(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int = 0
    responsibilities: list[str] = []
    technologies: list[str] = []


class EducationItem(BaseModel):
    institution: str = ""
    degree: str | None = None
    field: str | None = None
    year: int | None = None


class CertificationItem(BaseModel):
    name: str = ""
    issuer: str | None = None
    year: int | None = None
    expiry: int | None = None


class LanguageItem(BaseModel):
    language: str = ""
    proficiency: Literal["basic", "conversational", "fluent", "native"] = "basic"


class KeyProjectItem(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] | None = None
    url: str | None = None


class PublicationItem(BaseModel):
    title: str = ""
    venue: str | None = None
    year: int | None = None
    url: str | None = None


class AwardItem(BaseModel):
    title: str = ""
    issuer: str | None = None
    year: int | None = None


class Miscellaneous(BaseModel):
    certifications: list[CertificationItem] = []
    languages: list[LanguageItem] = []
    key_projects: list[KeyProjectItem] = []
    publications: list[PublicationItem] = []
    awards: list[AwardItem] = []
    other: list[str] = []


class CandidateProfile(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[SkillItem] = []
    experience: list[ExperienceItem] = []
    education: list[EducationItem] = []
    total_experience_years: float = 0.0
    miscellaneous: Miscellaneous = Miscellaneous()


class CandidateProfileModified(BaseModel):
    skills: list[SkillItem] = []
    experience: list[ExperienceItem] = []
    education: list[EducationItem] = []
    total_experience_years: float = 0.0
    miscellaneous: Miscellaneous = Miscellaneous()


class RequirementMapping(BaseModel):
    requirement: str = ""
    category: Literal["skills", "experience", "education", "certification", "key_projects", "publications", "other"] = "other"
    status: Literal["met", "partial", "not_met"] = "not_met"
    evidence: str | None = None
    notes: str | None = None


class StrengthItem(BaseModel):
    topic: str = ""
    description: str = ""
    evidence: str = ""


class GapItem(BaseModel):
    topic: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    description: str = ""
    bridgeable: bool = True


class SkillsAnalysisModified(BaseModel):
    analysis: str = ""
 
 
class ExperienceAnalysisModified(BaseModel):
    experience_summary: str = ""
    strengths: list[str] = []
    gaps: list[str] = []
 
 
class EducationAnalysisModified(BaseModel):
    education_summary: str = ""
    remarks: list[str] = []


class ResumeFitAnalysisModified(BaseModel):
    skills_analysis: SkillsAnalysisModified = SkillsAnalysisModified()
    experience_analysis: ExperienceAnalysisModified = ExperienceAnalysisModified()
    education_analysis: EducationAnalysisModified = EducationAnalysisModified()


class CriteriaRatings(BaseModel):
    degree_match: Literal["strong", "partial", "none"] = "none"
    experience: Literal["strong", "partial", "none"] = "none"
    skill_match: Literal["strong", "partial", "none"] = "none"
    projects: Literal["strong", "partial", "none"] = "none"
    internship: Literal["strong", "partial", "none"] = "none"
    certifications: Literal["strong", "partial", "none"] = "none"
    resume_quality: Literal["strong", "partial", "none"] = "none"


class CriterionScore(BaseModel):
    rating: Literal["strong", "partial", "none"] = "none"
    score: float = 0.0
    max: float = 0.0


class ResumeBreakdown(BaseModel):
    degree_match: CriterionScore = CriterionScore()
    experience: CriterionScore = CriterionScore()
    skill_match: CriterionScore = CriterionScore()
    projects: CriterionScore = CriterionScore()
    internship: CriterionScore = CriterionScore()
    certifications: CriterionScore = CriterionScore()
    resume_quality: CriterionScore = CriterionScore()


class FitAnalysis(BaseModel):
    resume_score: float = 0.0
    resume_weighted: float = 0.0
    resume_breakdown: ResumeBreakdown = ResumeBreakdown()
    requirements_mapping: list[RequirementMapping] = []
    strengths: list[StrengthItem] = []
    gaps: list[GapItem] = []
    resume_fit_analysis: ResumeFitAnalysisModified = ResumeFitAnalysisModified()
    fit_summary: str = ""
    overall_summary: str = ""
    criteria_ratings: CriteriaRatings = CriteriaRatings()


class OverviewAI(BaseModel):
    current_role: str = ""
    education: str = ""
    top_skills: list[str] = []
    experience_years: str = ""
    skills_matched: str = ""
    resume_fit_summary: str = ""
    core_strengths: list[str] = []
    critical_gaps: list[str] = []


class SkillsAnalysis(BaseModel):
    percentage: float = 0.0
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    analysis: str = ""
 
 
class ExperienceAnalysis(BaseModel):
    percentage: float = 0.0
    experience_summary: str = ""
    strengths: list[str] = []
    gaps: list[str] = []
 
 
class EducationAnalysis(BaseModel):
    percentage: float = 0.0
    education_summary: str = ""
    remarks: list[str] = []


class ResponsibilityMatch(BaseModel):
    responsibility: str
    matched: bool


class ResumeFitAnalysisAI(BaseModel):
    skills_analysis: SkillsAnalysis = SkillsAnalysis()
    experience_analysis: ExperienceAnalysis = ExperienceAnalysis()
    education_analysis: EducationAnalysis = EducationAnalysis()
    responsibility_match: list[ResponsibilityMatch] = []
    overall_assessment: str = ""


class ResumeAnalyzeResponse(BaseModel):
    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin_url: str | None = None
    resume_fit_score: float = 0.0
    overview_ai: OverviewAI = OverviewAI()
    resume_fit_analysis_ai: ResumeFitAnalysisAI = ResumeFitAnalysisAI()
    candidate_profile: CandidateProfileModified = CandidateProfileModified()
    fit_analysis: FitAnalysis = FitAnalysis()


class ResumeItem(BaseModel):
    candidate_id: int | None = None
    resume_s3_url: str | None = None
 
 
class BatchResumeAnalyzeRequest(BaseModel):
    jd_id: int | None = None
    resumes: list[ResumeItem] = []
    benchmark_s3_url: str | None = None
 
 
class BatchResumeAnalyzeResponse(BaseModel):
    data: list[ResumeAnalyzeResponse] = []
