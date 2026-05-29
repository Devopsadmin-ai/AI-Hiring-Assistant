from app.orchestrator.scoring import compute_resume_score, compute_resume_weighted
from app.core.logger import setup_logger
from app.llm.base import LLMClient
from app.utils import extract_json
from datetime import datetime

logger = setup_logger(__name__)

_SYSTEM_PROMPT = """
You are a strict resume analysis engine.
Your output feeds directly into deterministic Python scoring logic - accuracy and completeness are critical.
Extract ONLY what is explicitly stated in the resume. Do NOT infer, fabricate or assume.
The job description may be structured (bullet points, numbered lists) or unstructured (paragraphs). Parse all formats.
The resume may be any format - traditional, functional, academic, freshers, senior executive and so on. Handle all.
Every field marked with instructions must be populated.

Return STRICT JSON with EXACTLY the following top-level keys:

{
    "candidate_profile": {
        "name": "Candidate's full name in title case. Return empty string if not found.",
        "email": "Valid email address. Return empty string if not found.",
        "phone": "Valid phone number in international format (e.g., +919876543210, +14155552671, +971501234567). Always include country code. Do not include spaces or special characters. Return empty string if not found.",
        "skills": [
            {
                "name": "Skill name in title case. Extract ALL explicitly mentioned skills from the entire resume including skills sections, experience bullets, projects, certifications, technologies and tools used.",
                "category": "Strictly one of: technical, soft, domain. Do not generate any other value.",
                "proficiency": "Strictly one of: beginner, intermediate, advanced, expert. Do not generate any other value. Infer from context - years of use, seniority of role, explicit mention."
            }
        ],
        "experience": [
            {
                "company": "Company name in title case.",
                "title": "Job title in title case.",
                "start_date": "Start date in YYYY-MM format. Use null if not found.",
                "end_date": "End date in YYYY-MM format. Use null if not found or if currently working.",
                "duration_months": "Return 0.",
                "responsibilities": ["Start each with a strong action verb. Be specific - include metrics, technologies, outcomes where present in resume."],
                "technologies": ["Technology names in title case. Avoid duplicates."]
            }
        ],
        "education": [
            {
                "institution": "Institution name in title case.",
                "degree": "Degree name in title case. Use null if not specified.",
                "field": "Field of study in title case. Use null if not specified.",
                "year": "Graduation year (YYYY). Use null if not found."
            }
        ],
        "total_experience_years": "Return 0.0.",
        "miscellaneous": {
            "certifications": [
                {
                    "name": "Certification name in title case.",
                    "issuer": "Organization name in title case. Use null if not specified.",
                    "year": "Year obtained (YYYY). Use null if not found.",
                    "expiry": "Expiry year (YYYY). Use null if not found or if not applicable."
                }
            ],
            "languages": [
                {
                    "language": "Spoken or written human language name in title case.",
                    "proficiency": "Strictly one of: basic, conversational, fluent, native. Do not generate any other value."
                }
            ],
            "key_projects": [
                {
                    "name": "Project name in title case.",
                    "description": "One concise sentence, maximum 25 words. Include what was built and its impact.",
                    "technologies": ["Technology names in title case. Avoid duplicates."],
                    "url": "Extract valid URL from the resume content. Use null if not found."
                }
            ],
            "publications": [
                {
                    "title": "Publication title in title case.",
                    "venue": "Journal or conference name. Use null if not found.",
                    "year": "Year obtained (YYYY). Use null if not found.",
                    "url": "Extract valid URL from the resume content. Use null if not found."
                }
            ],
            "awards": [
                {
                    "title": "Award name in title case.",
                    "issuer": "Organization name in title case. Use null if not specified.",
                    "year": "Year obtained (YYYY). Use null if not found."
                }
            ],
            "other": ["Any relevant information not captured above - volunteer work, open source, patents and so on."]
        }
    },
    "requirements_mapping": [
        {
            "requirement": "EXACT text of one atomic requirement from the job description. Do not paraphrase or combine multiple requirements into one entry.",
            "category": "Strictly one of: skills, experience, education, certification, key_projects, publications, other. Do not generate any other value.",
            "status": "Strictly one of: met, partial, not_met. Do not generate any other value.",
            "evidence": "Specific evidence from the resume supporting the status. Use exact phrases where possible. Use null if no evidence.",
            "notes": "Brief explanation (maximum 25 words) justifying the status. Use null if not needed."
        }
    ],
    "strengths": [
        {
            "topic": "Short label in title case.",
            "description": "Concise description (maximum 25 words).",
            "evidence": "Specific supporting evidence from the resume."
        }
    ],
    "gaps": [
        {
            "topic": "Short label in title case.",
            "severity": "Strictly one of: low, medium, high. Do not generate any other value.",
            "description": "Concise description of the gap (maximum 25 words).",
            "bridgeable": "Boolean. True if gap can be learned quickly (<2 months), else false."
        }
    ],
    "resume_fit_analysis": {
        "skills_analysis": {
            "analysis": "2-3 sentences narrative on skill alignment with the job description."
        },
        "experience_analysis": {
            "experience_summary": "2-3 sentences summarising candidate total experience and relevance to this role.",
            "strengths": ["Specific experience strengths relevant to this job description."],
            "gaps": ["Specific experience gaps relevant to this job description."]
        },
        "education_analysis": {
            "education_summary": "1-3 sentences on highest qualification and its relevance to this role.",
            "remarks": ["Specific remarks about the educational background."]
        }
    },
    "fit_summary": "Concise narrative (3-5 sentences) explaining how well the candidate aligns with the role. Include strongest matching skills, relevant experience and domain expertise. Mention alignment with the company or role requirements and any notable strengths. Briefly highlight key gaps or risks if present. End with an overall assessment of suitability.",
    "overall_summary": "Exactly 4 sentences. Sentence 1: strongest fit area with specific skills and years. Sentence 2: most relevant role, company and tenure match. Sentence 3: key gap with impact. Sentence 4: clear hire or no-hire recommendation with confidence level.",
    "criteria_ratings": {
        "degree_match": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "experience": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "skill_match": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "projects": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "internship": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "certifications": "Strictly one of: strong, partial, none. Do not generate any other value.",
        "resume_quality": "Strictly one of: strong, partial, none. Do not generate any other value."
    }
}

REQUIREMENTS_MAPPING - CRITICAL RULES:

- This is the most important section. It feeds the scoring engine directly.
- Coverage: Map EVERY requirement from the job description - required AND preferred or nice-to-have.
    - Required qualifications → Map each one individually.
    - Preferred qualifications → Map each one individually.
    - Do NOT skip anything. Missing entries = missing scores.
- Atomicity: One requirement = one entry. Never combine.
    - Wrong: "Python, Django and REST APIs" → one entry.
    - Right: "Python" → entry 1, "Django" → entry 2, "REST APIs" → entry 3.
    - Wrong: "5+ years of backend development experience" + "microservices knowledge" → one entry.
    - Right: "5+ years of backend development experience" → entry 1, "Microservices knowledge" → entry 2.
- Categories - assign strictly:
    - skills: Programming languages, frameworks, tools, technologies, methodologies.
    - experience: Years of experience, domain experience, industry background, role-level experience.
    - education: Degree requirements, field of study, academic qualifications.
    - certification: Specific certifications, licences, accreditations.
    - key_projects: Specific project types or portfolio requirements.
    - publications: Research papers, patents, publications.
    - other: Soft skills, team size, travel, language requirements.
- Status - assign strictly:
    - met: Requirement is clearly and fully satisfied by resume evidence.
    - partial: Requirement is partially satisfied - some evidence but not complete.
    - not_met: No credible evidence in resume for this requirement. 
- Evidence - be specific:
    - Right: "Python listed under skills, used at specific companies or organizations".
    - Right: "M.Sc. Artificial Intelligence and Machine Learning, ARM College of Technology".
    - Wrong: "Candidate has Python" ← too vague.
    - Use null only when genuinely zero evidence exists.

SKILLS EXTRACTION - COMPREHENSIVE RULES:
 
- Extract skills from ALL sections of the resume:
    - Dedicated skills section (primary source).
    - Experience bullet points (secondary source - technologies used).
    - Projects section (tools and frameworks).
    - Certifications (implies knowledge of that domain).
    - Education (relevant technical subjects if explicitly mentioned).
- Do NOT duplicate the same skill. Consolidate variants:
    - "Python 3", "Python 3.9", "Python (advanced)" → "Python".
    - "React.js", "ReactJS", "React" → "React".
    - "AWS S3", "AWS EC2", "AWS Lambda" → "AWS" (unless each is individually significant).

STRENGTHS AND GAPS - RULES:
 
- strengths: 3-6 items. Must be:
    - Specific to THIS candidate (not generic praise).
    - Directly relevant to THIS job description.
    - Backed by concrete resume evidence (company, role, metric, project).
- gaps: 2-4 items. Must be:
    - Clearly missing from the resume relative to job description requirements.
    - Specific - name the skill, experience or qualification.
    - severity:
        - high → Critical for the role - candidate cannot perform core duties without it.
        - medium → Important but role is possible without it.
        - low → Nice to have, minimal impact on performance.
    - bridgeable: true if gap can realistically be closed in <2 months, false if requires years.

CRITICAL - TWO SEPARATE RATING SYSTEMS. DO NOT MIX THEM:

- requirements_mapping[].status → ONLY allowed values: met, partial, not_met.
- criteria_ratings[key] → ONLY allowed values: strong, partial, none.
- NEVER put "strong" or "none" in requirements_mapping.status.
- NEVER put "met" or "not_met" in criteria_ratings.

CRITERIA RATINGS DETAIL (APPLY STRICTLY):

- degree_match:
    - strong → Degree directly matches job description requirement (e.g., CS degree for software role).
    - partial → Related but not exact field (e.g., Physics degree for data science role).
    - none → Unrelated degree or no degree mentioned.
- experience:
    - strong → Meets or exceeds required years in job description.
    - partial → Within 1 year below the minimum requirement.
    - none → More than 1 year below requirement or no relevant experience.
- skill_match:
    - strong → ≥70% of job description required skills present.
    - partial → 30-69% of job description required skills present.
    - none → <30% of job description required skills present.
- projects:
    - strong → Clearly relevant domain projects present.
    - partial → Only tangential or loosely related projects.
    - none → No relevant projects.
- internship:
    - strong → Relevant domain internship present.
    - partial → Internship in adjacent or related domain.
    - none → No internship.
- certifications:
    - strong → Role-specific certifications present (e.g., AWS for cloud role, CFA for finance).
    - partial → Only general or tangential certifications.
    - none → No certifications.
- resume_quality:
    - strong → Well-structured, consistent formatting, clear sections, professional language, no clutter.
    - partial → Minor formatting inconsistencies or clarity issues.
    - none → Poorly structured, inconsistent, hard to parse or overly generic.
"""


def _safe_list(val):
    return val if isinstance(val, list) else []


def _safe_optional_str(val):
    if not isinstance(val, str):
        return None

    val = val.strip()

    return val if val else None


def _safe_year(val):
    if val is None:
        return None

    if isinstance(val, int):
        return val

    if isinstance(val, str):
        val = val.strip()

        if val.isdigit() and len(val) == 4:
            return int(val)

    return None


def _sanitize_year_fields(items, fields=("year",)):
    cleaned = []

    for item in _safe_list(items):
        if not isinstance(item, dict):
            continue

        item = item.copy()

        for field in fields:
            if field in item:
                item[field] = _safe_year(item.get(field))

        cleaned.append(item)

    return cleaned


def _months_between(start: str, end: str | None) -> int:
    if not start:
        return 0

    try:
        start_dt = datetime.strptime(start, "%Y-%m")
    
    except Exception:
        return 0

    if end:
        try:
            end_dt = datetime.strptime(end, "%Y-%m")
        
        except Exception:
            return 0
    
    else:
        end_dt = datetime.utcnow()

    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1

    return max(months, 0)


def _build_user_message(jd_input: dict, resume_input: dict, benchmark_input: dict | None = None) -> dict:
    current_date = datetime.utcnow().strftime("%Y-%m")
    
    content_blocks = []

    content_blocks.append(
        {
            "type": "text",
            "text": f"""SYSTEM CONTEXT:\n\nToday's date is {current_date}. When calculating experience duration_months for roles where end_date is null, assume the person is currently working and calculate duration from start_date until {current_date}.\n\n"""
        }
    )

    if jd_input["mode"] == "file":
        content_blocks.append(
            {
                "type": "file",
                "data": jd_input["data"],
                "mime_type": jd_input["mime_type"]
            }
        )

        content_blocks.append(
            {
                "type": "text",
                "text": "\n\nThe document above is the JOB DESCRIPTION.\n\n"
            }
        )

    else:
        content_blocks.append(
            {
                "type": "text",
                "text": f"JOB DESCRIPTION:\n\n{jd_input['content']}\n\n"
            }
        )

    if resume_input["mode"] == "file":
        content_blocks.append(
            {
                "type": "file",
                "data": resume_input["data"],
                "mime_type": resume_input["mime_type"]
            }
        )

        content_blocks.append(
            {
                "type": "text",
                "text": "\n\nThe document above is the CANDIDATE RESUME.\n\n"
            }
        )

    else:
        content_blocks.append(
            {
                "type": "text",
                "text": f"CANDIDATE RESUME:\n\n{resume_input['content']}\n\n"
            }
        )

    if benchmark_input is not None:
        if benchmark_input["mode"] == "file":
            content_blocks.append(
                {
                    "type": "file",
                    "data": benchmark_input["data"],
                    "mime_type": benchmark_input["mime_type"]
                }
            )

            content_blocks.append(
                {
                    "type": "text",
                    "text": ("\n\nThe document above is the BENCHMARK RESUME (an example of an ideal candidate). Use it as a reference when assessing how the candidate compares to the ideal profile.\n\n")
                }
            )

        else:
            content_blocks.append(
                {
                    "type": "text",
                    "text": (f"BENCHMARK RESUME (ideal candidate reference):\n\n{benchmark_input['content']}\n\n")
                }
            )

    if benchmark_input is not None:
        closing = (
            "Analyse the candidate resume against the job description. "
            "Use the BENCHMARK RESUME as a reference for what an ideal candidate looks like - "
            "compare the candidate's skills, experience and qualifications against it when assessing strengths, gaps and criteria_ratings. "
            "Return the JSON now."
        )
    
    else:
        closing = "Analyse the candidate resume against the job description. Return the JSON now."
 
    content_blocks.append({"type": "text", "text": closing})

    return {"role": "user", "content": content_blocks}


async def run_resume_analysis(
    llm: LLMClient,
    jd_input: dict,
    resume_input: dict,
    benchmark_input: dict | None = None
) -> tuple[dict, dict]:
    logger.info("Starting resume analysis")
    
    user_message = _build_user_message(jd_input, resume_input, benchmark_input)

    raw = await llm.complete_json(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_message
        ],
        temperature=0.0,
        max_tokens=8192
    )

    logger.info(f"LLM response received successfully ({len(raw)} characters)")

    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response for resume analysis.")

    try:
        data = extract_json(raw)

    except ValueError as e:
        logger.error(f"JSON parsing failed : {e}")
        raise ValueError(f"LLM returned invalid JSON : {e}")

    if not data:
        raise ValueError("LLM returned empty JSON object.")

    cp_raw = data.get("candidate_profile") or {}
    req_map = _safe_list(data.get("requirements_mapping"))
    strengths = _safe_list(data.get("strengths"))
    gaps = _safe_list(data.get("gaps"))
    resume_fit_analysis = data.get("resume_fit_analysis") or {}
    fit_summary = _safe_optional_str(data.get("fit_summary")) or ""
    overall_summary = _safe_optional_str(data.get("overall_summary")) or ""

    SKILL_ALIASES = {
        "soft",
        "soft_skills",
        "technical",
        "tech",
        "skill"
    }

    for r in req_map:
        raw_cat = (r.get("category") or "").strip().lower()

        if raw_cat in SKILL_ALIASES:
            r["category"] = "skills"

    misc_raw = cp_raw.get("miscellaneous") or {}

    miscellaneous = {
        "certifications": _sanitize_year_fields(misc_raw.get("certifications"), ("year", "expiry")),
        "languages": _safe_list(misc_raw.get("languages")),
        "key_projects": _safe_list(misc_raw.get("key_projects")),
        "publications": _sanitize_year_fields(misc_raw.get("publications"), ("year",)),
        "awards": _sanitize_year_fields(misc_raw.get("awards"), ("year",)),
        "other": _safe_list(misc_raw.get("other"))
    }

    experience_clean = []
    
    total_months = 0

    for exp in _safe_list(cp_raw.get("experience")):
        start = exp.get("start_date")
        end = exp.get("end_date")

        duration = _months_between(start, end)

        exp["duration_months"] = duration
        title = (exp.get("title") or "").lower()

        if duration and "intern" not in title:
            total_months += duration

        experience_clean.append(exp)

    total_experience_years = round(total_months / 12, 2)

    criteria_ratings = data.get("criteria_ratings")

    if not isinstance(criteria_ratings, dict):
        criteria_ratings = {}
    
    if not isinstance(resume_fit_analysis, dict):
        resume_fit_analysis = {
            "skills_analysis": {
                "analysis": ""
            },
            "experience_analysis": {
                "experience_summary": "",
                "strengths": [],
                "gaps": []
            },
            "education_analysis": {
                "education_summary": "",
                "remarks": []
            }
        }
    
    resume_score, resume_breakdown = compute_resume_score(criteria_ratings)
    resume_weighted = compute_resume_weighted(resume_score)

    education_clean = _sanitize_year_fields(cp_raw.get("education"), ("year",))
    
    candidate_profile = {
        "name": _safe_optional_str(cp_raw.get("name")),
        "email": _safe_optional_str(cp_raw.get("email")),
        "phone": _safe_optional_str(cp_raw.get("phone")),
        "skills": _safe_list(cp_raw.get("skills")),
        "experience": experience_clean,
        "education": education_clean,
        "total_experience_years": total_experience_years,
        "miscellaneous": miscellaneous
    }

    fit_analysis = {
        "resume_score": resume_score,
        "resume_weighted": resume_weighted,
        "resume_breakdown": resume_breakdown,
        "requirements_mapping": req_map,
        "strengths": strengths,
        "gaps": gaps,
        "resume_fit_analysis": resume_fit_analysis,
        "fit_summary": fit_summary,
        "overall_summary": overall_summary,
        "criteria_ratings": criteria_ratings
    }

    logger.info(f"Resume analysis completed successfully (resume_score = {resume_score}, resume_weighted = {resume_weighted})")

    return candidate_profile, fit_analysis
