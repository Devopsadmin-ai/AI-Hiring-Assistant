FINAL_WEIGHTS = {
    "resume": 0.25,
    "risk": 0.15,
    "interview": 0.30,
    "coding": 0.30
}

RESUME_CRITERIA = {
    "degree_match": 20.0,
    "experience": 20.0,
    "skill_match": 20.0,
    "projects": 15.0,
    "internship": 10.0,
    "certifications": 10.0,
    "resume_quality": 5.0
}

RISK_CRITERIA = {
    "resume_consistency": 20.0,
    "job_stability": 15.0,
    "skill_authenticity": 20.0,
    "ai_detection": 15.0,
    "employment_gaps": 10.0,
    "profile_validation": 10.0,
    "communication_authenticity": 10.0
}

INTERVIEW_CRITERIA = {
    "conceptual_understanding": 25.0,
    "problem_solving": 20.0,
    "depth_of_answers": 15.0,
    "communication_clarity": 15.0,
    "domain_knowledge": 15.0,
    "confidence_structure": 10.0
}

MATCH_MULTIPLIER = {
    "strong": 1.0,
    "partial": 0.5,
    "none": 0.0
}

JOB_STABILITY_MIN_MONTHS = 18

AI_SCORE_HUMAN_THRESHOLD = 0.2
AI_SCORE_ASSISTED_THRESHOLD = 0.5
AI_SCORE_STRONG_THRESHOLD = 0.8

SIGNIFICANT_GAP_MONTHS = 3


def get_final_recommendation(final_score: float) -> str:
    if final_score >= 75.0:
        return "hire"

    if final_score >= 50.0:
        return "consider"

    return "reject"


def compute_resume_score(criteria_ratings: dict) -> tuple[float, dict]:
    breakdown = {}
    total = 0.0

    for criterion, max_pts in RESUME_CRITERIA.items():
        rating = criteria_ratings.get(criterion, "none")
        multiplier = MATCH_MULTIPLIER.get(rating, 0.0)
        pts = round(max_pts * multiplier, 1)
        breakdown[criterion] = {"rating": rating, "score": pts, "max": max_pts}
        total += pts
    
    total_score = round(min(100.0, max(0.0, total)), 1)

    return total_score, breakdown


def compute_resume_weighted(resume_score: float) -> float:
    return round(resume_score * FINAL_WEIGHTS["resume"], 2)


def compute_job_stability_score(avg_tenure_months: int) -> float:
    if avg_tenure_months >= JOB_STABILITY_MIN_MONTHS:
        return 15.0
    
    if avg_tenure_months >= 12:
        return 10.0
    
    if avg_tenure_months >= 6:
        return 5.0
    
    return 0.0


def compute_ai_detection_score(ai_usage_score: float) -> float:
    if ai_usage_score <= AI_SCORE_HUMAN_THRESHOLD:
        return 15.0
    
    if ai_usage_score <= AI_SCORE_ASSISTED_THRESHOLD:
        return 10.0
    
    if ai_usage_score <= AI_SCORE_STRONG_THRESHOLD:
        return 5.0
    
    return 0.0


def compute_employment_gaps_score(gap_periods: list[dict]) -> float:
    significant = [g for g in gap_periods if g.get("duration_months", 0) > SIGNIFICANT_GAP_MONTHS]
    total_gap_months = sum(g.get("duration_months", 0) for g in significant)

    if total_gap_months == 0:
        return 10.0

    if total_gap_months <= 6:
        return 7.5

    if total_gap_months <= 12:
        return 5.0
    
    if total_gap_months <= 18:
        return 2.5

    return 0.0


def compute_risk_score(
    avg_tenure_months: int,
    gap_periods: list[dict],
    ai_usage_score: float,
    resume_consistency_score: float,
    skill_authenticity_score: float,
    profile_validation_score: float,
    communication_authenticity_score: float
) -> tuple[float, str, dict]:
    breakdown = {}

    breakdown["resume_consistency"] = {
        "score": min(resume_consistency_score, RISK_CRITERIA["resume_consistency"]),
        "max": RISK_CRITERIA["resume_consistency"]
    }
    
    breakdown["job_stability"] = {
        "score": min(compute_job_stability_score(avg_tenure_months), RISK_CRITERIA["job_stability"]),
        "max": RISK_CRITERIA["job_stability"]
    }
    
    breakdown["skill_authenticity"] = {
        "score": min(skill_authenticity_score, RISK_CRITERIA["skill_authenticity"]),
        "max": RISK_CRITERIA["skill_authenticity"]
    }
    
    breakdown["ai_detection"] = {
        "score": min(compute_ai_detection_score(ai_usage_score), RISK_CRITERIA["ai_detection"]),
        "max": RISK_CRITERIA["ai_detection"]
    }
    
    breakdown["employment_gaps"] = {
        "score": min(compute_employment_gaps_score(gap_periods), RISK_CRITERIA["employment_gaps"]),
        "max": RISK_CRITERIA["employment_gaps"]
    }
    
    breakdown["profile_validation"] = {
        "score": min(profile_validation_score, RISK_CRITERIA["profile_validation"]),
        "max": RISK_CRITERIA["profile_validation"]
    }
    
    breakdown["communication_authenticity"] = {
        "score": min(communication_authenticity_score, RISK_CRITERIA["communication_authenticity"]),
        "max": RISK_CRITERIA["communication_authenticity"]
    }

    safety_score = round(min(100.0, max(0.0, sum(v["score"] for v in breakdown.values()))), 1)

    if safety_score >= 70.0:
        risk_flag = "low"
    
    elif safety_score >= 40.0:
        risk_flag = "medium"
    
    else:
        risk_flag = "high"

    return safety_score, risk_flag, breakdown


def compute_risk_weighted(risk_score: float) -> float:
    return round(risk_score * FINAL_WEIGHTS["risk"], 2)


def compute_interview_score(dimension_scores: dict) -> tuple[float, dict]:
    breakdown = {}
    total = 0.0

    for criterion, max_pts in INTERVIEW_CRITERIA.items():
        score = min(dimension_scores.get(criterion, 0.0), max_pts)
        breakdown[criterion] = {"score": score, "max": max_pts}
        total += score

    total_score = round(min(100.0, max(0.0, total)), 1)

    return total_score, breakdown


def compute_interview_weighted(interview_score: float) -> float:
    return round(interview_score * FINAL_WEIGHTS["interview"], 2)
