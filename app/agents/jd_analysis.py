from app.core.logger import setup_logger
from app.llm.base import LLMClient
from app.utils import extract_json

logger = setup_logger(__name__)

_SYSTEM_PROMPT = """
You are a strict job description (JD) analysis engine.
Extract and structure information ONLY from the provided JD content.
Do NOT fabricate, infer, assume, summarize beyond the source or introduce missing information.
Every extracted item MUST be directly traceable to explicit JD content.
Prefer over-extraction to under-extraction when information is explicitly present.

Return STRICT JSON with EXACTLY the following top-level keys:

{
    "summary": "string",
    "key_responsibilities": ["string"],
    "preferred_qualifications": ["string"],
    "required_skills": ["string"]
}

FIELD DEFINITIONS:

1. summary

- Extract the job summary strictly from the JD.
- If an explicit summary, objective, about-role or profile-overview section exists:
    - Preserve the original meaning and wording as closely as possible.
    - Do NOT unnecessarily paraphrase or rewrite.
    - Clean only formatting or grammatical inconsistencies if needed.
- If no explicit summary exists:
    - Generate a concise summary strictly from explicit JD content.
- Include when explicitly present:
    - Core purpose of the role.
    - Functional scope.
    - Main responsibilities.
    - Ideal candidate profile.
    - Required experience.
    - Required education.
    - Required expertise.
    - Seniority expectations.
    - Collaboration expectations.
    - Leadership or mentoring expectations.
- Use 2-6 sentences as needed.
- Do NOT omit important information for brevity.
- Do NOT add interpretive or promotional language.

2. key_responsibilities

- Extract ALL explicitly stated responsibilities from the JD.
- Do NOT limit the number of responsibilities.
- Include responsibilities from:
    - Responsibilities sections.
    - Job summary sections.
    - Requirements sections.
    - Preferred qualification sections if responsibility-related.
    - Bullet points.
    - Paragraphs.
    - Tables.
    - Poorly formatted text.
- Each item MUST:
    - Start with a strong action verb where possible.
    - Be specific and implementation-focused.
    - Be derived ONLY from explicit JD content.
    - Be a single concise sentence or phrase.
    - Preserve meaningful implementation details.
- Merge only exact or obvious duplicates.
- Do NOT generalize specific responsibilities.
- Do NOT fabricate responsibilities.

3. preferred_qualifications

- Extract ALL explicitly mentioned:
    - Preferred qualifications.
    - Optional qualifications.
    - Desirable qualifications.
    - Bonus qualifications.
    - Good-to-have qualifications.
    - Added advantages.
- Include:
    - Preferred experience.
    - Bonus technologies.
    - Optional tools or frameworks.
    - Certifications.
    - Educational preferences.
    - Preferred domains.
    - Nice-to-have competencies.
    - Optional methodologies.
    - Additional qualifications.
- If explicitly stated mandatory qualifications are present and there is no dedicated required qualifications field:
    - Include them in preferred_qualifications while preserving their mandatory nature.
- Include all explicitly stated education, experience, certification, eligibility and qualification requirements here, including both mandatory and preferred qualifications.
- Extract ONLY from explicit JD content.
- Merge only exact or obvious duplicates.
- Do NOT infer missing qualifications.

4. required_skills

- Use short noun phrases only.
- Do NOT use full sentences.
- Extract ALL explicitly mentioned or clearly required:
    - Programming languages.
    - Frameworks and libraries.
    - Tools and platforms.
    - Cloud technologies.
    - DevOps technologies.
    - Databases.
    - APIs.
    - Testing tools.
    - Testing types.
    - QA concepts.
    - Methodologies.
    - Practices.
    - Certifications.
    - Architectural concepts.
    - Protocols.
    - Domain knowledge.
    - Technical competencies.
    - Soft skills ONLY if explicitly stated.
- Extract skills from ALL sections including:
    - Responsibilities.
    - Qualifications.
    - Preferred qualifications.
    - Competencies.
    - Requirements.
    - Technology stacks.
    - Bullet points.
    - Tables.
    - Paragraphs.
    - Poorly structured text.
- Extract skills even if they appear:
    - In grouped form.
    - In slash-separated form.
    - In comma-separated lists.
    - Inside sentences.
    - Inside brackets.
    - Inside responsibilities.
- Split grouped technologies where appropriate:
    - "Java/Python" → ["Java", "Python"].
    - "AWS, Azure or GCP" → ["AWS", "Azure", "Google Cloud"].
- Preserve meaningful multi-word skills:
    - "Machine Learning".
    - "REST APIs".
    - "CI/CD Integration".
    - "Cross-functional Collaboration".
- Do NOT derive abstract competencies unless explicitly stated.
- Do NOT infer skills from job titles or responsibilities alone.

NORMALIZATION RULES:

- Normalize aliases and abbreviations:
    - "JS" → "JavaScript".
    - "TS" → "TypeScript".
    - "GCP" → "Google Cloud".
    - "Github" → "GitHub".
- Remove unnecessary suffixes:
    - "programming".
    - "development".
    - "engineering".
- Examples:
    - "Python programming" → "Python".
    - "Java development" → "Java".
- Preserve official product or framework names exactly:
    - "GitHub Actions".
    - "Rest Assured".
    - "Spring Boot".

DEDUPLICATION RULES:

- Remove exact duplicates.
- Remove normalized duplicates.
- Keep the most industry-standard representation.
- Preserve unique variants carrying meaningful differences.

FORMATTING RULES:

- Use Title Case where appropriate.
- Preserve acronyms in uppercase:
    - API.
    - SQL.
    - AWS.
    - CI/CD.
    - REST.

IMPORTANT GLOBAL RULES:

- Extract strictly from explicit JD content.
- Maximize extraction coverage.
- Preserve granular details even if they appear repetitive.
- Do NOT omit valid information for brevity.
- Do NOT fabricate or hallucinate information.
- Normalize formatting while preserving meaning.
- Prefer completeness with precision.
- Maintain production-ready formatting.
- Avoid unnecessary summarization.
- Avoid redundancy only when information is truly duplicated.
- Reconstruct poorly formatted JD content cleanly without changing meaning.

EDGE CASE HANDLING:

- If responsibilities are sparse:
    - Return all available responsibilities without padding.
- If qualifications are sparse:
    - Return all available qualifications without padding.
- If skills are sparse:
    - Return all available skills without padding.
- If sections overlap:
    - Extract all meaningful unique information.
- If formatting is inconsistent:
    - Normalize formatting while preserving meaning.

STRICT OUTPUT REQUIREMENTS:

- Return ONLY valid JSON.
- Do NOT wrap JSON in markdown.
- Do NOT include explanations.
- Do NOT include commentary.
- Do NOT include additional keys.
- No trailing commas.
- Keys MUST match EXACTLY:
    - summary.
    - key_responsibilities.
    - preferred_qualifications.
    - required_skills.
"""


async def run_jd_analysis(
    llm: LLMClient,
    jd_input: dict,
    job_role: str | None = None,
    industry: str | None = None,
    employment_type: str | None = None,
    seniority_level: str | None = None,
    experience: str | None = None
) -> dict:
    logger.info("Starting job description analysis")
    
    meta_parts = []
    
    if job_role:
        meta_parts.append(f"Job Role: {job_role}")

    if industry:
        meta_parts.append(f"Industry: {industry}")

    if employment_type:
        meta_parts.append(f"Employment Type: {employment_type}")

    if seniority_level:
        meta_parts.append(f"Seniority Level: {seniority_level}")

    if experience:
        meta_parts.append(f"Experience: {experience}")

    meta_header = "\n".join(meta_parts)
    
    content_blocks = []

    mode = jd_input["mode"]

    if mode == "combined":
        supp = jd_input["supplementary"]

        content_blocks.append(
            {
                "type": "text",
                "text": (
                    f"JOB METADATA:\n\n{meta_header}\n\n"
                    f"PRIMARY JOB DESCRIPTION:\n\n{jd_input['primary_text']}\n\n"
                    f"SUPPLEMENTARY JOB DESCRIPTION FILE:\n\n"
                )
            }
        )

        content_blocks.append(
            {
                "type": "file",
                "data": supp["data"],
                "mime_type": supp["mime_type"]
            }
        )

        content_blocks.append(
            {
                "type": "text",
                "text": (
                    "\n\nAnalyse BOTH inputs together as a single combined job description. "
                    "Return the JSON now."
                )
            }
        )

    elif mode == "file":
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
                "text": (
                    f"\n\nThe document above is the JOB DESCRIPTION.\n\n"
                    f"JOB METADATA:\n\n{meta_header}\n\n"
                    f"Analyse the JD and return the JSON now."
                )
            }
        )

    else:
        content_blocks.append(
            {
                "type": "text",
                "text": (
                    f"JOB METADATA:\n\n{meta_header}\n\n"
                    f"JOB DESCRIPTION:\n\n{jd_input['content']}\n\n"
                    f"Analyse the JD and return the JSON now."
                )
            }
        )

    user_message = {"role": "user", "content": content_blocks}

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
        raise ValueError("LLM returned empty response for job description analysis.")

    try:
        data = extract_json(raw)

    except ValueError as e:
        logger.error(f"JSON parsing failed : {e}")
        raise ValueError(f"LLM returned invalid JSON : {e}")
    
    if not data:
        raise ValueError("LLM returned empty JSON object.")

    result = {
        "summary": data.get("summary") or "",
        "key_responsibilities": data.get("key_responsibilities") or [],
        "preferred_qualifications": data.get("preferred_qualifications") or [],
        "required_skills": data.get("required_skills") or []
    }
    
    logger.info(f"Job description analysis completed successfully (key_responsibilities = {len(result['key_responsibilities'])}, preferred_qualifications = {len(result['preferred_qualifications'])}, required_skills = {len(result['required_skills'])})")

    return result
