import os
import time
import asyncio
import tempfile
import subprocess
import json as _json
from app.api.schemas.resume_analysis import BatchResumeAnalyzeRequest, BatchResumeAnalyzeResponse, CandidateProfile, EducationAnalysis, ExperienceAnalysis, FitAnalysis, OverviewAI, ResponsibilityMatch, ResumeAnalyzeRequest, ResumeAnalyzeResponse, ResumeFitAnalysisAI, ResumeItem, SkillsAnalysis
from app.integrations.frontend_api import _download_file, _extract_jd_text, _file_input, _strip_html, extract_linkedin_from_bytes, extract_risk_inputs, fetch_candidate, fetch_job, transcript_segments_to_text
from app.api.schemas.interview_plan import CodingQuestionItem, FocusArea, InterviewPlan, InterviewPlanRequest, InterviewPlanResponse, QuestionCategory, StandardQuestionItem
from app.api.schemas.interview_analysis import EvaluationSummary, InterviewAnalysisItem, InterviewAnalyticalRequest, InterviewAnalyticalResponse, InterviewResult
from app.api.schemas.risk_analysis import CriterionScore, RiskAnalysisAI, RiskAnalyzeRequest, RiskAnalyzeResponse, RiskBreakdown, RiskLevelItem, RiskRatings
from app.integrations.apify import linkedin_profile_to_text, scrape_linkedin_profile
from app.api.schemas.jd_analysis import JDAnalyzeRequest, JDAnalyzeResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.agents.interview_analysis import run_interview_analysis
from app.agents.resume_analysis import run_resume_analysis
from app.agents.interview_plan import run_interview_plan
from app.agents.risk_analysis import run_risk_analysis
from fastapi import APIRouter, Depends, HTTPException
from app.agents.jd_analysis import run_jd_analysis
from app.llm.factory import get_llm_client
from app.core.logger import setup_logger
from app.core.config import settings
from app.llm.base import LLMClient
from urllib.parse import urlparse

logger = setup_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/api/v1")

ALLOWED_S3_EXTENSIONS = (".pdf", ".docx", ".json")

MAX_BATCH_SIZE = 10


def _err(code: str, message: str) -> dict:
    return {
        "error": code,
        "message": message
    }


def _check_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> None:
    api_key = settings.BACKEND_API_KEY

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=_err("MISCONFIGURED", "BACKEND_API_KEY is not set on the server.")
        )

    if not credentials or credentials.credentials != api_key:
        raise HTTPException(
            status_code=401,
            detail=_err("UNAUTHORIZED", "Missing or invalid Bearer token.")
        )
    

def is_valid_linkedin_url(url: str) -> bool:
    if not url:
        return False

    url = url.strip()

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    return (domain == "linkedin.com" or domain == "www.linkedin.com" or domain.endswith(".linkedin.com"))


def _docx_to_pdf_bytes(data: bytes) -> bytes:
    logger.info("Converting .docx to .pdf file")

    with tempfile.TemporaryDirectory() as tmpdir:
        docx_path = os.path.join(tmpdir, "input.docx")
        pdf_path = os.path.join(tmpdir, "input.pdf")

        with open(docx_path, "wb") as f:
            f.write(data)

        try:
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmpdir, docx_path],
                capture_output=True,
                timeout=30
            )
        
        except FileNotFoundError:
            raise RuntimeError("LibreOffice is not installed.")
        
        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice .docx to .pdf conversion timed out after 30 seconds.")

        if result.returncode != 0 or not os.path.exists(pdf_path):
            raise RuntimeError(f"LibreOffice conversion failed : {result.stderr.decode().strip()}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        logger.info(f"File converted successfully ({len(pdf_bytes)} bytes)")

        return pdf_bytes


def _file_to_main_input(data: bytes, filename: str) -> dict:
    if filename.lower().endswith(".pdf"):
        logger.info("File is in .pdf format, sending to LLM")
        return _file_input(data, filename)

    try:
        logger.info("File is in .docx format, converting before sending to LLM")
        pdf_bytes = _docx_to_pdf_bytes(data)
        pdf_filename = filename.rsplit(".", 1)[0] + ".pdf"
        return _file_input(pdf_bytes, pdf_filename)
    
    except RuntimeError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE_JD", str(e))
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE_JD", f".docx to .pdf conversion failed : {str(e)}")
        )


def _validate_s3_url(url: str) -> None:
    clean_url = url.split("?")[0].lower()

    if not any(clean_url.endswith(ext) for ext in ALLOWED_S3_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=_err("INVALID_S3_URL", "s3_url must point to a .pdf, .docx or .json file.")
        )


def _build_resume_response(candidate_profile: dict, fit_analysis: dict, job_data: dict, pre_extracted_linkedin: str | None = None) -> ResumeAnalyzeResponse:
    EXCLUDED_FIELDS = {
        "name",
        "email",
        "phone"
    }

    candidate_profile_modified = {k: v for k, v in candidate_profile.items() if k not in EXCLUDED_FIELDS}

    experience = candidate_profile.get("experience", [])

    if experience:
        latest = experience[0]
        current_role = f"{latest.get('title', '')} @ {latest.get('company', '')}".strip(" @")
    
    else:
        current_role = ""
 
    education_list = candidate_profile.get("education", [])

    if education_list:
        edu = education_list[0]
        edu_parts = [p for p in [edu.get("degree"), edu.get("field"), edu.get("institution")] if p]
        education_str = ", ".join(edu_parts)
    
    else:
        education_str = ""

    top_skills = [s.get("name", "") for s in candidate_profile.get("skills", [])[:10]]

    total_yrs = candidate_profile.get("total_experience_years", 0.0)
    experience_years = f"{round(total_yrs, 1)} Years"
 
    req_map = fit_analysis.get("requirements_mapping", [])
    matched_skills = [r["requirement"] for r in req_map if r.get("status") in ("met", "partial") and r.get("category") == "skills"]
    missing_skills = [r["requirement"] for r in req_map if r.get("status") == "not_met" and r.get("category") in ("skills", "certification")]
    total_skill_reqs = len([r for r in req_map if r.get("category") == "skills"])
    skills_matched_str = (f"{len(matched_skills)}/{total_skill_reqs}" if total_skill_reqs else f"{len(matched_skills)}/{len(matched_skills)}")

    core_strengths = [s.get("topic", "") for s in fit_analysis.get("strengths", [])]
    critical_gaps = [g.get("topic", "") for g in fit_analysis.get("gaps", [])]

    fit_score = fit_analysis.get("resume_score", 0.0)
 
    overview_ai = OverviewAI(
        current_role=current_role,
        education=education_str,
        top_skills=top_skills,
        experience_years=experience_years,
        skills_matched=skills_matched_str,
        resume_fit_summary=fit_analysis.get("fit_summary") or "",
        core_strengths=core_strengths,
        critical_gaps=critical_gaps
    )
 
    rfa = fit_analysis.get("resume_fit_analysis") or {}
    sa = rfa.get("skills_analysis") or {}
    ea = rfa.get("experience_analysis") or {}
    eda = rfa.get("education_analysis") or {}
 
    _rating_pct = {"strong": 100, "partial": 50, "none": 0}
    criteria = fit_analysis.get("criteria_ratings") or {}
 
    total_skills = len(matched_skills) + len(missing_skills)
    skills_pct = round(len(matched_skills) / total_skills * 100) if total_skills else 0
 
    required_min_yrs = job_data.get("experience_min_yr") or 0
    candidate_yrs = candidate_profile.get("total_experience_years") or 0.0
 
    if required_min_yrs and required_min_yrs > 0:
        experience_pct = min(100, round((candidate_yrs / required_min_yrs) * 100))
    
    else:
        experience_pct = _rating_pct.get(criteria.get("experience", "none"), 0)
 
    education_pct  = _rating_pct.get(criteria.get("degree_match", "none"), 0)
 
    resume_fit_analysis_ai = ResumeFitAnalysisAI(
        skills_analysis=SkillsAnalysis(
            percentage=skills_pct,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            analysis=sa.get("analysis", "")
        ),
        experience_analysis=ExperienceAnalysis(
            percentage=experience_pct,
            experience_summary=ea.get("experience_summary", ""),
            strengths=ea.get("strengths", []),
            gaps=ea.get("gaps", [])
        ),
        education_analysis=EducationAnalysis(
            percentage=education_pct,
            education_summary=eda.get("education_summary", ""),
            remarks=eda.get("remarks", [])
        ),
        responsibility_match=[
            ResponsibilityMatch(
                responsibility=r.get("requirement", ""),
                matched=r.get("status") in ("met", "partial")
            )
            for r in req_map if r.get("category") in ("experience", "skills")
        ],
        overall_assessment=fit_analysis.get("overall_summary", "")
    )
 
    return ResumeAnalyzeResponse(
        candidate_name=candidate_profile.get("name"),
        email=candidate_profile.get("email"),
        phone=candidate_profile.get("phone"),
        linkedin_url=pre_extracted_linkedin,
        resume_fit_score=fit_score,
        overview_ai=overview_ai,
        resume_fit_analysis_ai=resume_fit_analysis_ai,
        candidate_profile=candidate_profile_modified,
        fit_analysis=fit_analysis
    )


async def _resolve_jd_input(request: JDAnalyzeRequest) -> dict:
    has_text = bool(request.jd_text and request.jd_text.strip())
    has_s3 = bool(request.jd_s3_url)

    if has_text and has_s3:
        _validate_s3_url(request.jd_s3_url)

        try:
            data, filename = await _download_file(request.jd_s3_url)
        
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=_err("INVALID_S3_URL", str(e))
            )

        supp = _file_to_main_input(data, filename)

        return {
            "mode": "combined",
            "primary_text": request.jd_text.strip(),
            "supplementary": supp
        }

    if has_text:
        return {
            "mode": "text",
            "content": request.jd_text.strip()
        }

    _validate_s3_url(request.jd_s3_url)

    try:
        data, filename = await _download_file(request.jd_s3_url)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=_err("INVALID_S3_URL", str(e))
        )

    return _file_to_main_input(data, filename)


@router.post("/jd/analyze", response_model=JDAnalyzeResponse, summary="Analyse a job description", tags=["Step 1 — Job Description Analysis"])
async def jd_analyze(request: JDAnalyzeRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/jd/analyze' with job role '{request.job_role}'")
    
    start = time.time()

    if request.job_role is None or not isinstance(request.job_role, str) or not request.job_role.strip():
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (job_role) not provided.")
        )

    if (request.jd_text is None or not isinstance(request.jd_text, str) or not request.jd_text.strip()) and (request.jd_s3_url is None or not isinstance(request.jd_s3_url, str) or not request.jd_s3_url.strip()):
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Neither jd_text nor jd_s3_url was provided.")
        )

    try:
        if request.jd_text:
            request.jd_text = _strip_html(request.jd_text)

        jd_input = await _resolve_jd_input(request)
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE_JD", "The file/text could not be parsed into a valid JD.")
        )

    try:
        result = await run_jd_analysis(
            llm=llm,
            jd_input=jd_input,
            job_role=request.job_role,
            industry=request.industry,
            employment_type=request.employment_type,
            seniority_level=request.seniority_level,
            experience=request.experience
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    except Exception as e:
        logger.error(f"Job description analysis failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )

    end = time.time()

    logger.info("Closing endpoint '/jd/analyze'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return result


@router.post("/resume/analyze", response_model=ResumeAnalyzeResponse, summary="Analyse a candidate resume against a job description", tags=["Step 2 — Resume Analysis"])
async def resume_analyze(request: ResumeAnalyzeRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/resume/analyze' with job id '{request.jd_id}' and candidate id '{request.candidate_id}'")
    
    start = time.time()

    if request.jd_id is None or not isinstance(request.jd_id, int) or request.candidate_id is None or not isinstance(request.candidate_id, int) or request.resume_s3_url is None or not isinstance(request.resume_s3_url, str) or not request.resume_s3_url.strip():
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (jd_id, candidate_id, resume_s3_url) not provided.")
        )

    try:
        job_data = await fetch_job(request.jd_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)

        if "JD_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("JD_NOT_FOUND", msg.replace("JD_NOT_FOUND: ", ""))
            )

        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )

    try:
        jd_text = _extract_jd_text(job_data)
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    jd_input = {"mode": "text", "content": jd_text}

    _validate_s3_url(request.resume_s3_url)

    try:
        resume_bytes, resume_filename = await _download_file(request.resume_s3_url)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=_err("INVALID_S3_URL", str(e))
        )
    
    resume_input = _file_to_main_input(resume_bytes, resume_filename)

    benchmark_input = None
 
    if isinstance(request.benchmark_s3_url, str) and request.benchmark_s3_url.strip():
        _validate_s3_url(request.benchmark_s3_url)

        try:
            benchmark_bytes, benchmark_filename = await _download_file(request.benchmark_s3_url)
            benchmark_input = _file_to_main_input(benchmark_bytes, benchmark_filename)
            logger.info(f"Benchmark resume downloaded successfully")
        
        except Exception as e:
            logger.warning(f"Benchmark resume download failed : {e}")
    
    pre_extracted_linkedin = None

    if resume_filename.lower().endswith((".pdf", ".docx")):
        url = extract_linkedin_from_bytes(resume_bytes, resume_filename)
        pre_extracted_linkedin = url or None

    try:
        candidate_profile, fit_analysis = await run_resume_analysis(
            llm=llm,
            jd_input=jd_input,
            resume_input=resume_input,
            benchmark_input=benchmark_input
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    except Exception as e:
        logger.error(f"Resume analysis failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )

    try:
        cp_validated = CandidateProfile(**candidate_profile)
        fa_validated = FitAnalysis(**fit_analysis)
    
    except Exception as e:
        logger.error(f"Output validation failed : {e}")
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", f"Output validation failed : {str(e)}")
        )
    
    result = _build_resume_response(
        candidate_profile = cp_validated.model_dump(),
        fit_analysis = fa_validated.model_dump(),
        job_data = job_data,
        pre_extracted_linkedin = pre_extracted_linkedin
    )

    end = time.time()

    logger.info("Closing endpoint '/resume/analyze'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return result


@router.post("/resume/analyze/batch", response_model=BatchResumeAnalyzeResponse, summary="Analyse multiple candidate resumes in parallel against a job description", tags=["Step 2 — Resume Analysis"])
async def resume_analyze_batch(request: BatchResumeAnalyzeRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/resume/analyze/batch' with job id '{request.jd_id}' and resume count '{len(request.resumes)}'")
    
    start = time.time()

    if request.jd_id is None or not isinstance(request.jd_id, int) or request.resumes is None or not isinstance(request.resumes, list) or len(request.resumes) == 0:
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (jd_id, resumes) not provided.")
        )

    for resume in request.resumes:
        if resume.candidate_id is None or not isinstance(resume.candidate_id, int) or resume.resume_s3_url is None or not isinstance(resume.resume_s3_url, str) or not resume.resume_s3_url.strip():
            raise HTTPException(
                status_code=400,
                detail=_err("MISSING_INPUT", "Required fields (candidate_id, resume_s3_url) not provided.")
            )
 
    if len(request.resumes) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", f"Maximum {MAX_BATCH_SIZE} resumes per batch. Received {len(request.resumes)}.")
        )
 
    try:
        job_data = await fetch_job(request.jd_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)

        if "JD_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("JD_NOT_FOUND", msg.replace("JD_NOT_FOUND: ", ""))
            )

        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )
 
    try:
        jd_text = _extract_jd_text(job_data)
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
 
    jd_input = {"mode": "text", "content": jd_text}
 
    benchmark_input = None
 
    if isinstance(request.benchmark_s3_url, str) and request.benchmark_s3_url.strip():
        _validate_s3_url(request.benchmark_s3_url)

        try:
            benchmark_bytes, benchmark_filename = await _download_file(request.benchmark_s3_url)
            benchmark_input = _file_to_main_input(benchmark_bytes, benchmark_filename)
            logger.info(f"Benchmark resume downloaded successfully")
        
        except Exception as e:
            logger.warning(f"Benchmark resume download failed : {e}")
 
    async def _analyse_one(item: ResumeItem) -> ResumeAnalyzeResponse | None:
        _validate_s3_url(item.resume_s3_url)
        
        try:
            resume_bytes, resume_filename = await _download_file(item.resume_s3_url)
        
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=_err("INVALID_S3_URL", str(e))
            )
        
        resume_input = _file_to_main_input(resume_bytes, resume_filename)

        pre_extracted_linkedin = None

        if resume_filename.lower().endswith((".pdf", ".docx")):
            url = extract_linkedin_from_bytes(resume_bytes, resume_filename)
            pre_extracted_linkedin = url or None

        try:
            candidate_profile, fit_analysis = await run_resume_analysis(
                llm=llm,
                jd_input=jd_input,
                resume_input=resume_input,
                benchmark_input=benchmark_input
            )
        
        except HTTPException:
            raise
        
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=_err("UNPROCESSABLE", str(e))
            )
        
        except Exception as e:
            logger.error(f"Resume analysis failed : {e}")
            raise HTTPException(
                status_code=500,
                detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
            )

        try:
            cp_validated = CandidateProfile(**candidate_profile)
            fa_validated = FitAnalysis(**fit_analysis)
        
        except HTTPException:
            raise
        
        except Exception as e:
            logger.error(f"Output validation failed : {e}")
            raise HTTPException(
                status_code=422,
                detail=_err("UNPROCESSABLE", f"Output validation failed : {str(e)}")
            )

        response = _build_resume_response(
            candidate_profile = cp_validated.model_dump(),
            fit_analysis = fa_validated.model_dump(),
            job_data = job_data,
            pre_extracted_linkedin = pre_extracted_linkedin
        )

        logger.info(f"Batch candidate {item.candidate_id} completed successfully")
        
        return response
 
    try:
        raw_results = await asyncio.gather(*[_analyse_one(item) for item in request.resumes])
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Batch resume analysis failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )

    data = [r for r in raw_results if r is not None]

    logger.info(f"total = {len(request.resumes)}, successful = {len(data)}, failed = {len(request.resumes) - len(data)}")

    end = time.time()

    logger.info("Closing endpoint '/resume/analyze/batch'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return BatchResumeAnalyzeResponse(data=data)


@router.post("/interview/plan", response_model=InterviewPlanResponse, summary="Generate an interview plan for a candidate", tags=["Step 3 — Interview Plan"])
async def interview_plan(request: InterviewPlanRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/interview/plan' with job id '{request.job_id}' and candidate id '{request.candidate_id}'")
    
    start = time.time()

    if request.job_id is None or not isinstance(request.job_id, int) or request.candidate_id is None or not isinstance(request.candidate_id, int):
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (job_id or candidate_id) not provided.")
        )

    try:
        job_data = await fetch_job(request.job_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)

        if "JD_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("JOB_NOT_FOUND", f"No job found for job_id {request.job_id}.")
            )

        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )

    try:
        jd_text = _extract_jd_text(job_data)
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )

    try:
        candidate_data = await fetch_candidate(request.candidate_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)
        
        if "CANDIDATE_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("CANDIDATE_NOT_FOUND", f"No candidate found for candidate_id {request.candidate_id}.")
            )
        
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )

    ai_resume_misc = candidate_data.get("ai_resume_misc") or {}
    candidate_profile = ai_resume_misc.get("candidate_profile") or {}
    fit_analysis = ai_resume_misc.get("fit_analysis") or {}
    
    try:
        result = await run_interview_plan(
            llm=llm,
            jd_text=jd_text,
            candidate_profile=candidate_profile,
            fit_analysis=fit_analysis,
            question_min=request.question_min or 10,
            question_max=request.question_max or 15
        )
    
    except Exception as e:
        logger.error(f"Interview plan failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )

    raw_plan = result.get("interview_plan", {})

    focus_areas = [
        FocusArea(
            area=fa.get("area", ""),
            priority=fa.get("priority", "Medium"),
            reason=fa.get("reason", "")
        )
        for fa in raw_plan.get("interview_focus_areas", [])
    ]

    categories = [
        QuestionCategory(
            category=cat.get("category", ""),
            questions=[
                (
                    CodingQuestionItem(
                        id=q.get("id", i + 1),
                        title=q.get("title", ""),
                        task=q.get("task", ""),
                        focus_area=q.get("focus_area", ""),
                        difficulty=q.get("difficulty", "Medium"),
                        example=q.get("example", ""),
                        input=q.get("input", ""),
                        output=q.get("output", "")
                    )
                    if cat.get("category") == "Coding"
                    else StandardQuestionItem(
                        id=q.get("id", i + 1),
                        question=q.get("question", ""),
                        focus_area=q.get("focus_area", ""),
                        difficulty=q.get("difficulty", "Medium"),
                        hints=q.get("hints", ""),
                        possible_answers=q.get("possible_answers", [])
                    )
                )
                for i, q in enumerate(cat.get("questions", []))
            ]
        )
        for cat in raw_plan.get("question_categories", [])
    ]

    end = time.time()

    logger.info("Closing endpoint '/interview/plan'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return InterviewPlanResponse(
        interview_plan=InterviewPlan(
            interview_focus_areas=focus_areas,
            question_categories=categories
        )
    )


@router.post("/risk/analyze", response_model=RiskAnalyzeResponse, summary="Analyse a candidate for risk signals and profile consistency", tags=["Step 4 — Risk Analysis"])
async def risk_analyze(request: RiskAnalyzeRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/risk/analyze' with candidate id '{request.candidate_id}'")
    
    start = time.time()
 
    if request.candidate_id is None or not isinstance(request.candidate_id, int):
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (candidate_id) not provided.")
        )
 
    if request.linkedin_url is not None and (not isinstance(request.linkedin_url, str) or not is_valid_linkedin_url(request.linkedin_url)):
        raise HTTPException(
            status_code=400,
            detail=_err("INVALID_LINKEDIN_URL", "The LinkedIn profile URL is inaccessible or private.")
        )
    
    try:
        candidate_data = await fetch_candidate(request.candidate_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)
        
        if "CANDIDATE_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("CANDIDATE_NOT_FOUND", f"No candidate found for candidate_id {request.candidate_id}.")
            )
        
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )
    
    risk_inputs = extract_risk_inputs(candidate_data)
 
    if not risk_inputs["experience"]:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", "Candidate has no experience data.")
        )
 
    linkedin_text = ""
    linkedin_extraction_condition = ""

    linkedin_url = (request.linkedin_url or "").strip()
    
    if linkedin_url and settings.APIFY_API_KEY:
        try:
            profile = await scrape_linkedin_profile(linkedin_url, settings.APIFY_API_KEY)
            
            if profile:
                linkedin_extraction_condition = "extracted"
                linkedin_text = linkedin_profile_to_text(profile)
                logger.info(f"LinkedIn profile scraped successfully ({len(linkedin_text)} characters)")
            
            else:
                linkedin_extraction_condition = "not extracted"
                logger.warning(f"No data returned from Apify for {linkedin_url} - profile may be private or rate-limited, continuing without LinkedIn data.")
        
        except Exception as e:
            linkedin_extraction_condition = "not extracted"
            logger.warning(f"LinkedIn profile scrape failed : {e}")
    
    else:
        linkedin_extraction_condition = "not extracted"
        logger.warning("linkedin_url or APIFY_API_KEY not set, continuing risk analysis without LinkedIn data.")
 
    try:
        result = await run_risk_analysis(
            llm = llm,
            experience = risk_inputs["experience"],
            candidate_text = risk_inputs["candidate_text"],
            linkedin_text = linkedin_text
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    except Exception as e:
        logger.error(f"Risk analysis failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )
    
    rb = result["risk_breakdown"]

    end = time.time()

    logger.info("Closing endpoint '/risk/analyze'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return RiskAnalyzeResponse(
        risk_score=result["risk_score"],
        safety_score=result["safety_score"],
        risk_flag=result["risk_flag"],
        risk_weighted=result["risk_weighted"],
        linkedin_extraction_condition=linkedin_extraction_condition,
        risk_ratings=RiskRatings(**result["risk_ratings"]),
        risk_breakdown=RiskBreakdown(**{k: CriterionScore(score=v["score"], max=v["max"]) for k, v in rb.items()}),
        risk_analysis_ai=RiskAnalysisAI(
            overview=result["risk_analysis_ai"]["overview"],
            ai_risk_narrative=result["risk_analysis_ai"]["ai_risk_narrative"],
            core_strengths=result["risk_analysis_ai"]["core_strengths"],
            critical_gaps=result["risk_analysis_ai"]["critical_gaps"],
            overall_risk_level=[RiskLevelItem(**item) for item in result["risk_analysis_ai"]["overall_risk_level"]]
        )
    )


@router.post("/interview/analytical", response_model=InterviewAnalyticalResponse, summary="Analyse an interview transcript and evaluate candidate performance", tags=["Step 5 — Interview Analysis"])
async def interview_analytical(request: InterviewAnalyticalRequest, llm: LLMClient = Depends(get_llm_client), _auth: None = Depends(_check_auth)):
    logger.info(f"Calling endpoint '/interview/analytical' with job id '{request.jd_id}' and candidate id '{request.candidate_id}'")
    
    start = time.time()

    if request.jd_id is None or not isinstance(request.jd_id, int) or request.candidate_id is None or not isinstance(request.candidate_id, int) or request.transcript_s3_url is None or not isinstance(request.transcript_s3_url, str) or not request.transcript_s3_url.strip():
        raise HTTPException(
            status_code=400,
            detail=_err("MISSING_INPUT", "Required fields (jd_id, candidate_id, transcript_s3_url) not provided.")
        )
    
    try:
        job_data = await fetch_job(request.jd_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)

        if "JD_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("JD_NOT_FOUND", msg.replace("JD_NOT_FOUND: ", ""))
            )

        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )

    try:
        jd_text = _extract_jd_text(job_data)
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    try:
        candidate_data = await fetch_candidate(request.candidate_id, settings.FRONTEND_API_BASE, settings.FRONTEND_API_TOKEN)
    
    except ValueError as e:
        msg = str(e)
        
        if "CANDIDATE_NOT_FOUND" in msg:
            raise HTTPException(
                status_code=404,
                detail=_err("CANDIDATE_NOT_FOUND", f"No candidate found for candidate_id {request.candidate_id}.")
            )
        
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", msg)
        )

    ai_resume_misc = candidate_data.get("ai_resume_misc") or {}
    candidate_profile = ai_resume_misc.get("candidate_profile") or {}
    fit_analysis = ai_resume_misc.get("fit_analysis") or {}

    _validate_s3_url(request.transcript_s3_url)

    try:
        transcript_bytes, _ = await _download_file(request.transcript_s3_url)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=_err("INVALID_S3_URL", str(e))
        )

    try:
        transcript_json = _json.loads(transcript_bytes.decode("utf-8"))
        segments = transcript_json.get("segments", [])
        
        if not segments:
            raise HTTPException(
                status_code=422,
                detail=_err("UNPROCESSABLE", "Transcript has no segments.")
            )
        
        transcript_text = transcript_segments_to_text(segments)
        
        logger.info(f"Transcript parsed successfully ({len(segments)} segments, {len(transcript_text)} characters)")
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", f"Failed to parse transcript JSON : {str(e)}")
        )
    
    try:
        result, interview_score, interview_weighted, interview_breakdown = await run_interview_analysis(
            llm = llm,
            jd_text = jd_text,
            candidate_profile = candidate_profile,
            fit_analysis = fit_analysis,
            transcript_text = transcript_text
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=_err("UNPROCESSABLE", str(e))
        )
    
    except Exception as e:
        logger.error(f"Interview analysis failed : {e}")
        raise HTTPException(
            status_code=500,
            detail=_err("INTERNAL_ERROR", "Unexpected error on the AI service side.")
        )

    raw_analysis = result.get("interview_analysis", [])

    interview_analysis = [
        InterviewAnalysisItem(
            id=item.get("id", i + 1),
            question=item.get("question", ""),
            category=item.get("category", "Technical"),
            status=item.get("status", "partially_correct"),
            received_answer=item.get("received_answer", ""),
            answer_summary=item.get("answer_summary", ""),
            possible_answers=item.get("possible_answers", [])
        )
        for i, item in enumerate(raw_analysis)
    ]

    def breakdown_to_percentage(breakdown: dict) -> dict:
        return {
            key: round((value.get("score", 0.0) / max_) * 100, 0)
            if (max_ := value.get("max", 1.0))
            else 0.0
            for key, value in breakdown.items()
        }
    
    evaluation_scores = breakdown_to_percentage(interview_breakdown)

    interview_result = InterviewResult(
        overall_interview_score=interview_score,
        interview_weighted=interview_weighted,
        interview_breakdown=interview_breakdown,
        overview=result.get("overview", ""),
        evaluation_summary=EvaluationSummary(
            conceptual_understanding=evaluation_scores.get("conceptual_understanding", 0.0),
            problem_solving=evaluation_scores.get("problem_solving", 0.0),
            depth_of_answers=evaluation_scores.get("depth_of_answers", 0.0),
            communication_clarity=evaluation_scores.get("communication_clarity", 0.0),
            domain_knowledge=evaluation_scores.get("domain_knowledge", 0.0),
            confidence_structure=evaluation_scores.get("confidence_structure", 0.0)
        ),
        core_strengths=result.get("core_strengths", []),
        critical_gaps=result.get("critical_gaps", []),
        follow_up_questions=result.get("follow_up_questions", [])
    )

    end = time.time()

    logger.info("Closing endpoint '/interview/analytical'")
    logger.info(f"Duration : {end - start:.2f} seconds")

    return InterviewAnalyticalResponse(
        total_questions = len(interview_analysis),
        interview_analysis = interview_analysis,
        interview_result = interview_result
    )
