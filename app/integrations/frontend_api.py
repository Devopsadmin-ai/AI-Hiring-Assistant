import io
import re
import fitz
import html
import httpx
import zipfile
from app.core.logger import setup_logger
from lxml import etree

logger = setup_logger(__name__)

TIMEOUT_SECONDS = 30


def _strip_html(html_text: str) -> str:
    if not html_text:
        return ""
    
    text = re.sub(r"<(script|style).*?>.*?</\1>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    
    return text


def _enrich_jd_text(jd_text: str, job_data: dict) -> str:
    header_parts = []
    
    if job_data.get("name"):
        header_parts.append(f"Job Title: {job_data['name']}")
    
    if job_data.get("industry_name"):
        header_parts.append(f"Industry: {job_data['industry_name']}")
    
    if job_data.get("emp_type"):
        header_parts.append(f"Employment Type: {job_data['emp_type']}")
    
    if job_data.get("seniority_level"):
        header_parts.append(f"Seniority: {job_data['seniority_level']}")
    
    if job_data.get("experience"):
        header_parts.append(f"Experience Required: {job_data['experience']}")

    if header_parts:
        return "\n".join(header_parts) + "\n\n" + jd_text
        
    return jd_text


def _extract_jd_text(job_data: dict) -> str:
    if job_data.get("description_ai"):
        dai = job_data["description_ai"]
        
        if isinstance(dai, dict):
            parts = []
            
            if dai.get("summary"):
                parts.append(dai["summary"])
            
            responsibilities = dai.get("key_responsibilities") or []
            qualifications = dai.get("preferred_qualifications") or []
            skills = dai.get("required_skills") or []
            
            if responsibilities:
                parts.append("")
                parts.append("Key Responsibilities:")
                parts.append("")
                parts.extend(f"- {r}" for r in responsibilities)

            if qualifications:
                parts.append("")
                parts.append("Preferred Qualifications:")
                parts.append("")
                parts.extend(f"- {r}" for r in qualifications)

            if skills:
                parts.append("")
                parts.append("Required Skills:")
                parts.append("")
                parts.extend(f"- {s}" for s in skills)
            
            text = "\n".join(parts).strip()
        
        else:
            text = str(dai).strip()
        
        if text:
            logger.info(f"description_ai serialized successfully ({len(text)} characters)")
            return _enrich_jd_text(text, job_data)
        
    raise ValueError("No usable job description text found, description_ai is all null or empty.")


def _mime_from_filename(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
    return {
        "pdf":  "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }.get(ext, "application/pdf")


def _file_input(data: bytes, filename: str) -> dict:
    return {
        "mode": "file",
        "data": data,
        "filename": filename,
        "mime_type": _mime_from_filename(filename)
    }


def _ensure_https(url: str) -> str:
    url = url.strip()
    
    if url and not url.startswith("http"):
        url = "https://" + url
    
    return url


def _extract_linkedin_pdf(content: bytes) -> str:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                
                if "linkedin.com" in uri.lower():
                    return _ensure_https(uri)
    
    except Exception as e:
        logger.error(f"LinkedIn extraction in .pdf failed : {e}")
    
    return ""


def _extract_linkedin_docx(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            rel_files = [n for n in z.namelist() if n.endswith(".rels") and "document" in n]
            
            for rel_file in rel_files:
                tree = etree.fromstring(z.read(rel_file))
                
                for rel in tree:
                    target = rel.get("Target", "")
                    mode = rel.get("TargetMode", "")
                    
                    if mode == "External" and "linkedin.com" in target.lower():
                        return _ensure_https(target)

            doc_xml = z.read("word/document.xml")
            tree = etree.fromstring(doc_xml)
            all_text = " ".join(t for t in tree.itertext())

            patterns = [
                r"https?://(?:www\.)?linkedin\.com/in/[\w\-]+",
                r"(?:www\.)?linkedin\.com/in/[\w\-]+",
                r"linkedin\.com/pub/[\w\-/]+"
            ]

            for pattern in patterns:
                m = re.search(pattern, all_text, re.IGNORECASE)
                
                if m:
                    return _ensure_https(m.group(0))

            label = re.search(r"(?:linkedin|linked-in)\s*[:\-]?\s*([\w\.\/\-]+)", all_text, re.IGNORECASE)
            
            if label:
                raw = label.group(1).strip().rstrip("/")
                skip = {"profile", "com", "in", "linkedin", "www"}
                
                if raw and raw.lower() not in skip:
                    if "." not in raw and "/" not in raw:
                        return f"https://linkedin.com/in/{raw}"
                    
                    return _ensure_https(raw)

    except Exception as e:
        logger.error(f"LinkedIn extraction in .docx failed : {e}")
    
    return ""


def extract_linkedin_from_bytes(content: bytes, filename: str) -> str:
    fn = filename.lower()
    
    if fn.endswith(".pdf"):
        url = _extract_linkedin_pdf(content)
    
    elif fn.endswith(".docx"):
        url = _extract_linkedin_docx(content)
    
    else:
        url = ""
    
    if url:
        logger.info(f"LinkedIn extracted from hyperlinks : {url!r}")
    
    else:
        logger.info("LinkedIn not found in hyperlinks")
    
    return url


def extract_risk_inputs(candidate_data: dict) -> dict:
    overview_ai = candidate_data.get("overview_ai") or {}
    ai_resume_misc = candidate_data.get("ai_resume_misc") or {}    
    candidate_profile = ai_resume_misc.get("candidate_profile") or {}
    fit_analysis = ai_resume_misc.get("fit_analysis") or {}
    experience = candidate_profile.get("experience", [])
    skills = [s.get("name", "") for s in candidate_profile.get("skills", []) if s.get("name")]
 
    lines = []
 
    name = candidate_data.get("candidate_name", "")
    
    if name:
        lines.append(f"Candidate Name: {name}")
    
    if overview_ai.get("current_role"):
        lines.append(f"Current Role: {overview_ai['current_role']}")
    
    if overview_ai.get("experience_years"):
        lines.append(f"Total Experience: {overview_ai['experience_years']}")
    
    if skills:
        lines.append(f"Skills: {', '.join(skills)}")
 
    if experience:
        lines.append("\nExperience:\n")
        
        for exp in experience:
            end_dt = exp.get("end_date") or "Present"
            lines.append(f"- {exp.get('title', '')} at {exp.get('company', '')} ({exp.get('start_date', '')} to {end_dt})")
            
            for resp in exp.get("responsibilities", [])[:10]:
                lines.append(f"    - {resp}")
                
    education = overview_ai.get("education")

    if isinstance(education, str):
        lines.append(f"\nEducation: {education}")

    elif isinstance(education, list) and education:
        edu = education[0]

        if isinstance(edu, dict):
            lines.append(f"Education: {edu.get('degree', '')} {edu.get('field', '')} at {edu.get('institution', '')}")
    
    strengths = fit_analysis.get("strengths", [])
    
    if strengths:
        lines.append("Strengths: " + ", ".join(s.get("topic", "") for s in strengths))
 
    gaps = fit_analysis.get("gaps", [])
    
    if gaps:
        lines.append("Gaps: " + ", ".join(g.get("topic", "") for g in gaps))
 
    fit_summary = fit_analysis.get("fit_summary", "") or overview_ai.get("resume_fit_summary", "")
    
    if fit_summary:
        lines.append(f"Fit Summary: {fit_summary}")
 
    req_map = fit_analysis.get("requirements_mapping", [])
    
    if req_map:
        lines.append("\nRequirements Mapping:\n")
        
        for r in req_map[:20]:
            lines.append(f"- {r.get('requirement', '')}: {r.get('status', '')} ({r.get('evidence') or 'no evidence'})")
 
    certs = candidate_profile.get("miscellaneous", {}).get("certifications", [])
    
    if certs:
        lines.append("\nCertifications: " + ", ".join(c.get("name", "") for c in certs))
 
    return {
        "experience": experience,
        "candidate_text": "\n".join(lines)
    }


def transcript_segments_to_text(segments: list) -> str:
    if not segments:
        return ""

    lines = []
    buffer = []

    prev_speaker = None

    for seg in segments:
        speaker = (seg.get("speaker") or "Unknown").strip()
        text = (seg.get("text") or "").strip()

        if not text:
            continue

        if speaker == prev_speaker:
            buffer.append(text)
        
        else:
            if prev_speaker and buffer:
                lines.append(f"{prev_speaker}: {' '.join(buffer)}")
            
            prev_speaker = speaker
            buffer = [text]

    if prev_speaker and buffer:
        lines.append(f"{prev_speaker}: {' '.join(buffer)}")

    return "\n".join(lines)


async def _download_file(url: str) -> tuple[bytes, str]:
    logger.info(f"Downloading file : {url}")
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        resp = await client.get(url, follow_redirects=True)
        
        if resp.status_code != 200:
            raise ValueError("Failed to download file")
    
    filename = url.split("/")[-1].split("?")[0] or "Resume.pdf"
    logger.info(f"File downloaded successfully ({len(resp.content)} bytes)")
    return resp.content, filename


async def fetch_job(job_id: int, base_url: str, token: str) -> dict:
    logger.info(f"Fetching job : {job_id}")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=headers) as client:
        resp = await client.get(f"{base_url}/jobs/{job_id}")
    
    if resp.status_code == 404:
        raise ValueError(f"JD_NOT_FOUND: No Job Description found for jd_id {job_id}.")
    
    if resp.status_code != 200:
        raise ValueError(f"Job fetch failed")
    
    job_data = resp.json()["data"]
    logger.info("Job fetched successfully")
    return job_data


async def fetch_candidate(candidate_id: int, base_url: str, token: str) -> dict:
    logger.info(f"Fetching candidate : {candidate_id}")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=headers) as client:
        resp = await client.get(f"{base_url}/project-job-resumes/{candidate_id}")
    
    if resp.status_code == 404:
        raise ValueError(f"CANDIDATE_NOT_FOUND: No candidate found for candidate_id {candidate_id}.")
    
    if resp.status_code != 200:
        raise ValueError(f"Candidate fetch failed")
    
    candidate_data = resp.json()["data"]
    logger.info("Candidate fetched successfully")
    return candidate_data
