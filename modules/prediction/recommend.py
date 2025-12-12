def quality_category(score: float) -> str:
    if score >= 90:
        return "Exceptional"
    elif score >= 85:
        return "Specialty"
    elif score >= 80:
        return "Premium"
    else:
        return "Commercial"

def recommendation_from_category(cat: str) -> str:
    if cat == "Exceptional":
        return "Kopi ini sangat cocok untuk kompetisi atau pasar eksklusif."
    if cat == "Specialty":
        return "Kopi ini ideal untuk pasar specialty; roasting light–medium."
    if cat == "Premium":
        return "Kopi ini cocok untuk premium blend dan retail harian."
    return "Kopi ini untuk pasar komersial atau blend"
