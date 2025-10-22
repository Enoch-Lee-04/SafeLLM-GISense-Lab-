from PIL import Image
from pathlib import Path

# Note: Login is not needed for local testing
# from huggingface_hub import login
# login(token="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", add_to_git_credential=True)

# note the image is not provided in the prompt its included as part of the "processor"
prompt= """Analyze this street view image and provide a safety score from 1-10 (10 being safest). Only return the safety score."""

system_message = "You are an expert safety analyst. You are given a street view image and you need to analyze the image and provide a safety score from 1-10 (10 being safest)."

# Convert image to OAI messages       
def format_data(image):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },{
                            "type": "image",
                            "image": image,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The safety score is 8 out of 10."}],
                },
            ],
        }

# Load image using path relative to script location
image_path = Path(__file__).parent.parent.parent / "data" / "images" / "image" / "1.jpg"
image = Image.open(image_path)

# Convert image to OAI messages
formatted_image = format_data(image)

# Print concise formatted output
print("=" * 80)
print("FORMATTED MESSAGE STRUCTURE")
print("=" * 80)

for i, message in enumerate(formatted_image["messages"], 1):
    print(f"\n[Message {i}] Role: {message['role'].upper()}")
    for content_item in message['content']:
        if content_item['type'] == 'text':
            print(f"  └─ Text: {content_item['text']}")
        elif content_item['type'] == 'image':
            img = content_item['image']
            print(f"  └─ Image: {img.format} {img.mode} {img.size[0]}x{img.size[1]}")

print("\n" + "=" * 80)