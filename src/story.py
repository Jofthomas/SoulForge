"""
Generate a list of initial story elements.
"""
def generate_seed_story_elements(tgi_client):
    # Set prompt params
    sys_prompt = "You are a creative writer of science fiction."
    prompt = """
        I want to generate a video game with NPCs and their stories.
        The context is space and star travelling.
        Create me 5 different contexts (persona + context + its story).
        Don't make more than 10 sentences per one item.
        Delimit each story by <STORY> tag.
        Put a name before each story.
    """
    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    input_text = sys_format + user_format + assistant_format

    # Generate seed story elements
    response = tgi_client.generate(prompt=input_text, max_new_tokens=512, do_sample=True, temperature=0.5).generated_text
    response = response.split("<STORY>")
    for story_id, story in enumerate(response):
        # Save it to file
        with open(f"./data/{story_id}.txt", "a") as f:
            f.write(story + "\n")