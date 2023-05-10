import re
def normalize_to_usual_dialogue(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, r"\1:", text)
    return normalized_string


def remove_roles(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, '', text)
    return normalized_string
