def build_scene_prompt():
    return (
        "I will provide you with a video from the first-person perspective of a visually impaired person. "
        "Your task is to describe the key objects, obstacles, layout, and any hazards visible in this environment. "
        "The description needs to be based on the content of the images and the objective characteristics that the blind person cannot see. "
        "If the scene cannot be described, you can say I don't know."
    )

def build_guidance_prompt(goal):
    return (
        "I will provide you with a video from the first-person perspective of a visually impaired person. "
        f"Their goal is: {goal} "
        "Your task is to provide clear step-by-step instructions to help them achieve this goal safely, "
        "based on the content of the images and the objective characteristics that the blind person cannot see. "
        "Highlight any hazards or important landmarks. "
        "If the goal cannot be addressed, you can say I don't know."
    )

def build_qa_prompt(question):
    return (
        "I will provide you with a video from the first-person perspective of a visually impaired person. "
        "These questions are raised by the blind person in their current scene. "
        "Your task is to answer the following question based on the content of the images "
        "and the objective characteristics that the blind person cannot see. "
        "If the question cannot be answered, you can say I don't know. "
        f"The question is: {question}"
    )