from pydantic import BaseModel


class JDAnalyzeRequest(BaseModel):
    job_role: str | None = None
    industry: str | None = None
    employment_type: str | None = None
    seniority_level: str | None = None
    experience: str | None = None
    jd_text: str | None = None
    jd_s3_url: str | None = None


class JDAnalyzeResponse(BaseModel):
    summary: str = ""
    key_responsibilities: list[str] = []
    preferred_qualifications: list[str] = []
    required_skills: list[str] = []
