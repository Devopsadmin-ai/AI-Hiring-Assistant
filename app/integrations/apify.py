import re
import httpx
import asyncio
from app.core.logger import setup_logger

logger = setup_logger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = "harvestapi~linkedin-profile-scraper"

POLL_INTERVAL = 3
POLL_TIMEOUT = 120

_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3,  "Apr": 4,
    "May": 5,  "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9,  "Oct": 10, "Nov": 11, "Dec": 12
}


async def scrape_linkedin_profile(linkedin_url: str, apify_token: str) -> dict:
    logger.info(f"Starting LinkedIn profile scrape : {linkedin_url}")
    
    params = {"token": apify_token}

    async with httpx.AsyncClient(timeout=30) as client:
        run_input = {
            "profileScraperMode": "Profile details no email ($4 per 1k)",
            "queries": [linkedin_url.rstrip("/")],
            "urls": [],
            "publicIdentifiers": [],
            "profileIds": []
        }
        
        start_resp = await client.post(
            f"{APIFY_BASE}/acts/{ACTOR_ID}/runs",
            params=params,
            json=run_input
        )

        if start_resp.status_code not in (200, 201):
            logger.error(f"Failed to start actor run ({start_resp.status_code} | {start_resp.text})")
            return {}

        run_data = start_resp.json()["data"]
        run_id = run_data["id"]
        dataset_id = run_data["defaultDatasetId"]
        
        logger.info(f"Actor run started (run_id = {run_id})")

        elapsed = 0
        
        while elapsed < POLL_TIMEOUT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            status_resp = await client.get(f"{APIFY_BASE}/actor-runs/{run_id}", params=params)
            status = status_resp.json()["data"]["status"]
            
            logger.info(f"Actor run processing (status = {status}, elapsed = {elapsed} seconds)")

            if status == "SUCCEEDED":
                break
            
            if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                logger.error(f"Actor run {status} (run_id = {run_id})")
                return {}
        
        else:
            logger.error(f"Polling timed out after {POLL_TIMEOUT} seconds")
            return {}

        items_resp = await client.get(f"{APIFY_BASE}/datasets/{dataset_id}/items", params={**params, "format": "json", "clean": True})

        if items_resp.status_code != 200:
            logger.error(f"Dataset fetch failed ({items_resp.status_code})")
            return {}

        items = items_resp.json()
        
        if not items:
            logger.warning(f"No items returned for {linkedin_url}")
            return {}

        profile = items[0]

        return _normalise_profile(profile)


def _normalise_date(date_val) -> str | None:
    if not date_val or not isinstance(date_val, dict):
        return None
    
    year = date_val.get("year")
    month = date_val.get("month")
    
    if not year:
        return None
    
    month_num = _MONTH_MAP.get(month, 1) if isinstance(month, str) else (month or 1)
    
    return f"{year}-{int(month_num):02d}"


def _normalise_profile(raw: dict) -> dict:
    first = raw.get("firstName", "") or ""
    last = raw.get("lastName",  "") or ""
    name = f"{first} {last}".strip()
    location_raw = raw.get("location", {})
    
    if isinstance(location_raw, dict):
        location = location_raw.get("linkedinText", "") or location_raw.get("parsed", {}).get("text", "")
    
    else:
        location = str(location_raw) if location_raw else ""

    experiences = []
    
    for exp in raw.get("experience", []):
        experiences.append(
            {
                "company": exp.get("companyName", ""),
                "title": exp.get("position", ""),
                "start_date": _normalise_date(exp.get("startDate")),
                "end_date": _normalise_date(exp.get("endDate")),
                "employment_type": exp.get("employmentType", ""),
                "description": exp.get("description", "") or ""
            }
        )

    education = []
    
    for edu in raw.get("education", []):
        education.append(
            {
                "institution": edu.get("schoolName", ""),
                "degree": edu.get("degree", ""),
                "field": edu.get("fieldOfStudy", "") or "",
                "start_year": (edu.get("startDate") or {}).get("year"),
                "end_year": (edu.get("endDate")   or {}).get("year")
            }
        )

    skills = [s.get("name", "") for s in raw.get("skills", []) if isinstance(s, dict) and s.get("name")]
    
    certifications = []
    
    for c in raw.get("certifications", []):
        issued_at = c.get("issuedAt", "") or ""
        year = None
        
        m = re.search(r"\d{4}", issued_at)
        
        if m:
            year = int(m.group(0))
        
        certifications.append(
            {
                "name": c.get("title", ""),
                "issuer": c.get("issuedBy", ""),
                "year": year
            }
        )

    return {
        "name": name,
        "headline": raw.get("headline", ""),
        "summary": raw.get("about", "") or "",
        "location": location,
        "experience": experiences,
        "education": education,
        "skills": skills,
        "certifications": certifications
    }


def linkedin_profile_to_text(profile: dict) -> str:
    if not profile:
        return ""

    lines = []

    if profile.get("name"):
        lines.append(f"Name: {profile['name']}")
    
    if profile.get("headline"):
        lines.append(f"Headline: {profile['headline']}")
    
    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")
    
    if profile.get("summary"):
        lines.append(f"Summary: {profile['summary']}")

    if profile.get("experience"):
        lines.append("\nExperience:\n")
        
        for exp in profile["experience"]:
            start = exp.get("start_date") or ""
            end = exp.get("end_date") or "Present"
            emp_type = exp.get("employment_type", "")
            lines.append(f"- {exp.get('title', '')} at {exp.get('company', '')} ({start} to {end})" + (f" [{emp_type}]" if emp_type else ""))

    if profile.get("education"):
        lines.append("\nEducation:\n")
        
        for edu in profile["education"]:
            start_yr = edu.get("start_year", "")
            end_yr = edu.get("end_year", "")
            period = f"{start_yr} - {end_yr}" if start_yr else ""
            lines.append(f"- {edu.get('degree', '')} at {edu.get('institution', '')}" + (f" ({period})" if period else ""))

    if profile.get("skills"):
        lines.append(f"\nSkills: {', '.join(profile['skills'][:30])}")

    if profile.get("certifications"):
        lines.append("\nCertifications:\n")
        
        for cert in profile["certifications"]:
            year = f" ({cert['year']})" if cert.get("year") else ""
            lines.append(f"- {cert.get('name', '')} - {cert.get('issuer', '')}{year}")

    return "\n".join(lines)
