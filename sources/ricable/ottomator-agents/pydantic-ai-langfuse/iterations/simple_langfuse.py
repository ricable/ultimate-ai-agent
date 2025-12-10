from langfuse.decorators import observe
from langfuse.openai import openai
from dotenv import load_dotenv

load_dotenv(".env", override=True)

@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
          {"role": "system", "content": "You are a great storyteller."},
          {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
    ).choices[0].message.content
 
@observe()
def main():
    new_story = story()
    print(new_story)
    return new_story

if __name__ == "__main__":
    main()