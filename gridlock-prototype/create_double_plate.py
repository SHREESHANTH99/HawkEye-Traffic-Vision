from PIL import Image, ImageDraw, ImageFont

def create_double_plate():
    # Indian plate: White background, black text, double row
    width, height = 300, 200
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Top row
    draw.text((20, 20), "MH 12", fill="black", font=font)
    # Bottom row
    draw.text((20, 100), "AB 1234", fill="black", font=font)
    
    img.save("static/violations/double_plate.jpg")
    print("Created double_plate.jpg")

create_double_plate()
