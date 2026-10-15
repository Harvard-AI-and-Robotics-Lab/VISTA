def build_blind_prompt(question):
    """Blind-aware prompt from EgoBlind paper (Table 12)."""
    return (
        "I will provide you with a video each time and one question; "
        "These questions are all questions raised by the blind person in the video from his own first-person perspective in the current scene. "
        "Your task is to answer the blind person's question which was posed in the last frame of the video. "
        "The answer needs to be based on the content of the picture and the objective characteristics that the blind person cannot see. "
        "If the question cannot be answered, you can say I don't know. "
        "Do not include Chinese characters in your response. "
        f"The question is: {question}"
    )