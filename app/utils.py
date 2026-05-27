import re
import json


def extract_json(raw: str) -> dict:
    raw = raw.lstrip("\ufeff")
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().replace("```", "").strip()

    try:
        return json.loads(raw)
    
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    depth, in_string, escape = 0, False, False

    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        
        if ch == "\\" and in_string:
            escape = True
            continue
        
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if ch == "{":
            depth += 1
        
        elif ch == "}":
            depth -= 1

            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                
                except json.JSONDecodeError as e:
                    raise ValueError(f"Extracted JSON is invalid : {e}")

    raise ValueError("Could not find a complete JSON object in LLM response.")
