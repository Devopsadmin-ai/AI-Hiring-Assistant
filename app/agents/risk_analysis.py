from app.orchestrator.scoring import compute_risk_score, compute_risk_weighted, JOB_STABILITY_MIN_MONTHS, SIGNIFICANT_GAP_MONTHS
from app.core.logger import setup_logger
from app.llm.base import LLMClient
from app.utils import extract_json
from datetime import datetime

logger = setup_logger(__name__)

_SYSTEM_PROMPT = """
You are a strict risk analysis engine.
Analyse the candidate profile and their LinkedIn data.
Your output feeds downstream hiring-risk scoring systems.
Accuracy, conservatism and evidence-based reasoning are critical.

Return STRICT JSON with EXACTLY the following top-level keys:

{
    "overview": "2-4 sentences of plain-English overview of overall risk profile for this candidate.",
    "ai_risk_narrative": "4-8 sentences of factual narrative covering profile consistency, experience validation, skill alignment, communication authenticity and risk observations.",
    "core_strengths": ["Specific validated positive signals supported by evidences."],
    "critical_gaps": ["Specific discrepancy, inconsistency or missing validations requiring follow-ups."],
    "risk_ratings": {
        "resume_consistency": "Return 0.0.",
        "skill_authenticity": "Return 0.0.",
        "profile_validation": "Return 0.0.",
        "communication_authenticity": "Return 0.0."
    },
    "ai_usage_score": "Return 0.0."
}

IMPORTANT GLOBAL RULES:

- Use ONLY explicitly available evidence.
- Never fabricate risks, fraud indicators or inconsistencies.
- Never infer dishonesty without direct evidence.
- Absence of evidence is NOT evidence of fraud.
- If uncertain, prefer lower-risk conclusions.
- Conservative scoring is mandatory.

STRICT OUTPUT RULES:

- Output ONLY valid JSON.
- Do NOT wrap JSON in markdown.
- Do NOT include explanations outside JSON.
- Do NOT add extra keys.
- Do NOT omit required keys.
- Use [] for empty arrays.
- Do NOT use trailing commas.
- Scores must be numeric.
- ai_usage_score must be decimal between 0.0 and 1.0.

CORE ANALYSIS RULES:

- Analyse ONLY the following:
    - Timeline consistency.
    - Company or title consistency.
    - Skill validation across resume and LinkedIn.
    - Project consistency.
    - Education consistency.
    - Communication style consistency.
    - Signs of templated or AI-generated writing.
    - Missing profile completeness signals.
- Never analyse:
    - Personality.
    - Intelligence.
    - Ethics.
    - Protected attributes.
    - Demographic assumptions.
    - Nationality assumptions.
    - Age assumptions.
    - Political views.
    - Mental health.
    - Intent or motivation.

RISK ANALYSIS GUIDELINES:

- LOW RISK indicators:
    - Resume and LinkedIn timelines align.
    - Job titles match across sources.
    - Skills are supported by projects or work history.
    - Consistent career progression.
    - Detailed and specific project descriptions.
    - Natural variation in writing style.
    - Reasonable endorsement between skills and experience.
- MEDIUM RISK indicators:
    - Minor date mismatches.
    - Missing months in timelines.
    - Overly broad skill lists without supporting evidence.
    - Generic project descriptions.
    - Sparse LinkedIn profile sections.
    - Highly templated wording in some sections.
- HIGH RISK indicators:
    - Contradictory employment timelines.
    - Conflicting job titles for same role.
    - Major unexplained employment gaps.
    - Skills claimed with zero supporting evidence.
    - Duplicate or repetitive AI-style phrasing throughout.
    - Multiple inconsistent profile identities.
- Important:
    - Do NOT classify something as HIGH RISK unless explicit evidence exists.
    - Missing LinkedIn data alone is NOT high risk.
    - Generic resume language alone is NOT proof of AI generation.
    - AI-generated writing suspicion must remain probabilistic and conservative.

AI-GENERATED CONTENT ANALYSIS RULES:

- Evaluate communication authenticity conservatively.
- Possible AI-generated indicators:
    - Repetitive sentence structures.
    - Excessive buzzword density.
    - Generic achievement phrasing.
    - Unnaturally uniform tone.
    - Repeated template patterns.
    - Over-polished summaries lacking specifics.
- Human-authentic indicators:
    - Minor inconsistencies.
    - Specific contextual details.
    - Varied sentence structures.
    - Concrete metrics or examples.
    - Natural imperfections.
- Important:
    - AI assistance is common and NOT inherently negative.
    - Never claim content is definitely AI-generated.
    - Use probabilistic language only.
    - ai_usage_score represents likelihood of substantial AI assistance.
- ai_usage_score scale:
    - 0.0-0.2 → Strongly human-authored.
    - 0.3-0.5 → Possible AI-assisted polishing.
    - 0.6-0.8 → Strong AI-assisted writing indicators.
    - 0.9-1.0 → Highly templated or generated content patterns.

SCORING RULES:

- risk_ratings (LOW risk = HIGH score).
- resume_consistency (0.0-20.0):
    - 20.0 → Timelines, companies and roles fully consistent.
    - 15.0 → Very small inconsistencies.
    - 10.0 → Moderate inconsistencies needing clarification.
    - 5.0 → Major unexplained inconsistencies.
    - 0.0 → Clear contradictory evidence.
- skill_authenticity (0.0-20.0):
    - 20.0 → Skills consistently supported by projects or experience.
    - 15.0 → Most skills supported.
    - 10.0 → Several unsupported skills.
    - 5.0 → Many inflated or unsupported claims.
    - 0.0 → Clear evidence of fabricated skills.
- profile_validation (0.0-10.0):
    - 10.0 → LinkedIn strongly validates resume.
    - 7.5 → Mostly aligned with minor gaps.
    - 5.0 → Partial validation only.
    - 2.5 → Weak validation with multiple discrepancies.
    - 0.0 → Conflicting identities or major mismatch.
- communication_authenticity (0.0-10.0):
    - 10.0 → Natural, specific and human-authored communication.
    - 7.5 → Mostly natural with some templated phrasing.
    - 5.0 → Moderate AI-assisted or templated signals.
    - 2.5 → Heavily templated or generated style.
    - 0.0 → Extremely synthetic or repetitive content.

ANTI-HALLUCINATION RULES:

- Never invent companies, projects or skills.
- Never infer missing dates.
- Never assume employment gaps unless dates explicitly show them.
- Never infer AI generation from polished writing alone.
- Never infer fraud without direct contradiction.
- If evidence is weak, prefer moderate or low concern.
- If data is insufficient, state that validation is limited.

EMPTY DATA RULES:

- If LinkedIn data is missing:
    - Strictly assign zero score to profile_validation alone.
- If communication samples are unavailable:
    - Base communication_authenticity only on available text.
    - Avoid overconfident AI assessments.

FINAL SAFETY RULES:

- This system provides hiring-risk signals only.
- It must NOT make definitive fraud accusations or identity judgments.
- All findings must remain evidence-based, conservative and explainable.
"""


def _today_ym() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def _months_between(start: str, end: str) -> int:
    try:
        s = datetime.strptime(start.strip(), "%Y-%m")
        e = datetime.strptime(end.strip(), "%Y-%m")
        months = (e.year - s.year) * 12 + (e.month - s.month)
        return max(0, months - 1)

    except Exception:
        return 0


def _compute_gap_periods(experience: list) -> list:
    today = _today_ym()
    sorted_exp = sorted(experience, key=lambda r: r.get("start_date") or "0000-00")
    
    gaps = []
    
    for i in range(1, len(sorted_exp)):
        prev_end = sorted_exp[i - 1].get("end_date") or today
        curr_start = sorted_exp[i].get("start_date") or ""
        
        if not prev_end or not curr_start:
            continue
        
        gap_months = _months_between(prev_end, curr_start)
        
        if gap_months > 1:
            gaps.append({"start": prev_end, "end": curr_start, "duration_months": gap_months})
    
    return gaps


def _compute_tenure_metrics(experience: list) -> dict:
    if not experience:
        return {"avg_tenure_months": 0}

    durations = []

    for exp in experience:
        d = exp.get("duration_months")

        if d is not None:
            try:
                d = int(d)

                if d > 0:
                    durations.append(d)

            except Exception:
                pass

    avg = round(sum(durations) / len(durations)) if durations else 0

    return {"avg_tenure_months": avg}


def _resume_consistency_desc(pct: float, gap_periods: list) -> str:
    significant_gaps = [g for g in gap_periods if g.get("duration_months", 0) > SIGNIFICANT_GAP_MONTHS]
    gap_count = len(significant_gaps)

    gap_text = ""

    if gap_count == 1:
        gap_text = " 1 employment gap identified."

    elif gap_count > 1:
        gap_text = f" {gap_count} employment gaps identified."

    if pct >= 80:
        return "Profile information is highly consistent across roles and timelines." + gap_text

    if pct >= 60:
        return "Minor profile or timeline inconsistencies detected." + gap_text

    if pct >= 40:
        return "Moderate inconsistencies detected in experience or timeline data." + gap_text

    return "Significant profile inconsistencies detected." + gap_text


def _job_stability_desc(avg_tenure_months: int) -> str:
    if avg_tenure_months >= JOB_STABILITY_MIN_MONTHS:
        return (f"Strong job stability with an average tenure of {avg_tenure_months} months.")

    if avg_tenure_months >= 12:
        return (f"Moderate-to-strong job stability with an average tenure of {avg_tenure_months} months.")

    if avg_tenure_months >= 6:
        return (f"Moderate job stability with an average tenure of {avg_tenure_months} months.")

    return (f"Limited tenure stability observed with an average tenure of {avg_tenure_months} months.")


def _skill_authenticity_desc(pct: float) -> str:
    if pct >= 80:
        return ("Most listed skills are supported by professional experience and profile evidence.")

    if pct >= 60:
        return ("Several listed skills are supported by profile evidence though some areas have limited validation.")

    if pct >= 40:
        return ("Limited supporting evidence found for some listed skills within the profile data.")

    return ("Profile data provides minimal validation for several listed skills.")


def _ai_detection_desc(pct: float) -> str:
    ai_prob = round(100 - pct)

    if ai_prob <= 40:
        return (f"{ai_prob} % estimated AI assistance probability - content appears largely human-written.")

    if ai_prob <= 60:
        return (f"{ai_prob} % estimated AI assistance probability - moderate AI-assisted writing signals detected.")

    if ai_prob <= 80:
        return (f"{ai_prob} % estimated AI assistance probability - strong AI-assisted writing indicators detected.")

    return (f"{ai_prob} % estimated AI assistance probability - content may rely heavily on AI-generated phrasing.")


def _employment_gaps_desc(pct: float, gap_periods: list) -> str:
    significant_gaps = [g for g in gap_periods if g.get("duration_months", 0) > SIGNIFICANT_GAP_MONTHS]
    gap_count = len(significant_gaps)
    total_gap_months = sum(g.get("duration_months", 0) for g in significant_gaps)

    if gap_count == 0:
        return "No significant employment gaps detected."

    gap_text = (f"{gap_count} significant employment gap identified" if gap_count == 1 else f"{gap_count} significant employment gaps identified")

    if pct >= 80:
        return (f"{gap_text} totaling {total_gap_months} months - limited concern.")

    if pct >= 60:
        return (f"{gap_text} totaling {total_gap_months} months - moderate follow-up recommended.")

    return (f"{gap_text} totaling {total_gap_months} months - extended employment interruptions observed.")


def _profile_validation_desc(pct: float) -> str:
    if pct >= 80:
        return (f"LinkedIn profile shows strong alignment with resume data ({int(pct)} % match confidence).")

    if pct >= 60:
        return (f"LinkedIn profile partially aligns with resume data ({int(pct)} % match confidence), with minor inconsistencies detected.")

    return (f"Limited profile alignment detected ({int(pct)} % match confidence). Additional verification may be helpful.")


def _communication_authenticity_desc(pct: float) -> str:
    if pct >= 80:
        return ("Communication style appears natural and consistent across profile sections.")

    if pct >= 60:
        return ("Some profile sections appear standardized or templated in tone.")

    if pct >= 40:
        return ("Multiple sections exhibit repetitive or highly structured writing patterns.")

    return ("Communication patterns show strong indications of heavily assisted or template-based writing.")


async def run_risk_analysis(
    llm: LLMClient,
    experience: list,
    candidate_text: str,
    linkedin_text: str
) -> dict:
    logger.info("Starting risk analysis")

    content_blocks = [
        {
            "type": "text",
            "text": f"CANDIDATE PROFILE:\n\n{candidate_text}\n\n"
        },
        {
            "type": "text",
            "text": f"LINKEDIN PROFILE DATA:\n\n{linkedin_text}\n\n" if linkedin_text else "LINKEDIN PROFILE DATA: Not available.\n\n"
        },
        {
            "type": "text",
            "text": "Analyse and return the JSON now."
        }
    ]

    raw = await llm.complete_json(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content_blocks}
        ],
        temperature=0.0,
        max_tokens=8192
    )

    logger.info(f"LLM response received successfully ({len(raw)} characters)")

    if not raw or not raw.strip():
        raise ValueError("LLM returned empty response for risk analysis.")

    try:
        data = extract_json(raw)

    except ValueError as e:
        logger.error(f"JSON parsing failed : {e}")
        raise ValueError(f"LLM returned invalid JSON : {e}")

    if not data:
        raise ValueError("LLM returned empty JSON object.")

    gap_periods = _compute_gap_periods(experience)
    metrics = _compute_tenure_metrics(experience)
    ai_score = float(data.get("ai_usage_score", 0.0))
    risk_ratings = data.get("risk_ratings", {})

    safety_score, risk_flag, risk_breakdown = compute_risk_score(
        avg_tenure_months=metrics["avg_tenure_months"],
        gap_periods=gap_periods,
        ai_usage_score=ai_score,
        resume_consistency_score=float(risk_ratings.get("resume_consistency", 0.0)),
        skill_authenticity_score=float(risk_ratings.get("skill_authenticity", 0.0)),
        profile_validation_score=float(risk_ratings.get("profile_validation", 0.0)),
        communication_authenticity_score=float(risk_ratings.get("communication_authenticity", 0.0))
    )

    risk_weighted = compute_risk_weighted(safety_score)
    risk_score = round(100.0 - safety_score, 1)

    def _to_pct(key: str) -> float:
        item = risk_breakdown.get(key, {})
        score = item.get("score", 0.0)
        max_ = item.get("max", 1.0)
        return round(score / max_ * 100, 0) if max_ else 0.0

    overall_risk_level = [
        {
            "title": "Resume Consistency",
            "desc": _resume_consistency_desc(_to_pct("resume_consistency"), gap_periods),
            "percent": _to_pct("resume_consistency")
        },
        {
            "title": "Job Stability",
            "desc": _job_stability_desc(metrics["avg_tenure_months"]),
            "percent": _to_pct("job_stability")
        },
        {
            "title": "Skill Authenticity",
            "desc": _skill_authenticity_desc(_to_pct("skill_authenticity")),
            "percent": _to_pct("skill_authenticity")
        },
        {
            "title": "AI Detection",
            "desc": _ai_detection_desc(_to_pct("ai_detection")),
            "percent": _to_pct("ai_detection")
        },
        {
            "title": "Employment Gaps",
            "desc": _employment_gaps_desc(_to_pct("employment_gaps"), gap_periods),
            "percent": _to_pct("employment_gaps")
        },
        {
            "title": "Profile Validation",
            "desc": _profile_validation_desc(_to_pct("profile_validation")),
            "percent": _to_pct("profile_validation")
        },
        {
            "title": "Communication Authenticity",
            "desc": _communication_authenticity_desc(_to_pct("communication_authenticity")),
            "percent": _to_pct("communication_authenticity")
        }
    ]

    logger.info(f"Risk analysis completed successfully (risk_score = {risk_score}, safety_score = {safety_score}, risk_flag = {risk_flag}, risk_weighted = {risk_weighted})")

    return {
        "risk_score": risk_score,
        "safety_score": safety_score,
        "risk_flag": risk_flag,
        "risk_weighted": risk_weighted,
        "risk_ratings": {
            "resume_consistency": float(risk_ratings.get("resume_consistency", 0)),
            "skill_authenticity": float(risk_ratings.get("skill_authenticity", 0)),
            "profile_validation": float(risk_ratings.get("profile_validation", 0)),
            "communication_authenticity": float(risk_ratings.get("communication_authenticity", 0))
        },
        "risk_breakdown": risk_breakdown,
        "risk_analysis_ai": {
            "overview": data.get("overview", ""),
            "ai_risk_narrative": data.get("ai_risk_narrative", ""),
            "core_strengths": data.get("core_strengths", []),
            "critical_gaps": data.get("critical_gaps", []),
            "overall_risk_level": overall_risk_level
        }
    }
