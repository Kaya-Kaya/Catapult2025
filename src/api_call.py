import os
from google import genai
import requests
from dotenv import load_dotenv


def load_api_key():
    load_dotenv()
    GEMINI_KEY = os.getenv("GEMINI_KEY")

    if not(GEMINI_KEY):
        raise Exception("GEMINI_KEY not found in environment variables. Please set it up in a .env file.")
    
    genai.configure(api_key=GEMINI_KEY)

def prompt_gemini():
    prompt = """
    You are a golfing expert who understands the theory of optimal golf swing mechanics.
    You are also a golf coach who can explain the theory of optimal golf swing mechanics to a beginner golfer.
    Based on this information below, please provide a detailed and constructive feedback on the posture of a golfer, given a scoring of a few parameters.
    A higher score indicates a better posture.
    The scoring is based on the following metrics:
    Ball Position: Scored between 0.0-1.0
    -Description: The position of the ball in relation to the golfer's stance.
    Iron Stance: Scored between 0.0-1.0
    -Description: The stance of the golfer when using an iron club.
    Elbow Posture Backswing: Scored between 0.0-1.0
    -Description: The position of the elbows during the backswing.
    Elbow Posture Frontswing: Scored between 0.0-1.0
    -Description: The position of the elbows during the front swing.
    If the golfer is putting:
    Putting Stance: Scored between 0.0-1.0
    -Description: The stance of the golfer when putting.
    If the golfer is chipping:
    Chipping Stance: Scored between 0.0-1.0
    -Description: The stance of the golfer when chipping.
    """
    response = genai.generate_text(prompt=prompt)

    if response and "candidates" in response and len(response["candidates"]) > 0:
        # Assume the first candidate is what you want
        generated_text = response["candidates"][0]["output"]
        print("Generated Text:")
        print(generated_text)
    else:
        # If there was an error, the response might include an error message.
        error_message = response.get("error") if response else "No response received."
        print("Error generating text:", error_message)