import requests, base64, cv2
cap = cv2.VideoCapture('test_traffic.mp4')
ret, frame = cap.read()
cap.release()
_, buf = cv2.imencode('.jpg', frame)
b64 = base64.b64encode(buf.tobytes()).decode()
print('b64 len:', len(b64))
prompts = [
    'Describe what you see.',
    'What is in this image?',
    'Are there any cars visible?',
    'Is there a vehicle?',
    'Is the driver wearing a seatbelt?',
    'Look at this traffic image. Is any rider missing a helmet?',
]
for prompt in prompts:
    resp = requests.post('http://localhost:11434/api/generate', json={
        'model': 'moondream:latest',
        'prompt': prompt,
        'images': [b64],
        'stream': False,
    }, timeout=30)
    data = resp.json()
    ec  = data.get('eval_count', 0)
    raw = data.get('response', '')[:80]
    print(f"eval={ec:3}  prompt={repr(prompt[:40])}")
    print(f"       resp={repr(raw)}")
    print()
