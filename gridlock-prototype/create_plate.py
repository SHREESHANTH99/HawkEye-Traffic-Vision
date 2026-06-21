from PIL import Image, ImageDraw, ImageFont

def create_plate():
    # Indian plate: White background, black text, double row or single row. Let's do single row first.
    width, height = 400, 100
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    text = "MH 12 AB 1234"
    # Approximate centering
    draw.text((20, 20), text, fill="black", font=font)
    
    img.save("static/violations/clear_plate.jpg")
    print("Created clear_plate.jpg")

create_plate()
