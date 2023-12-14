import numpy as np
import cv2
import io

async def image_reader(file):
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image

def result_bytes(image):
    result_bytes = cv2.imencode(".png", image)[1].tobytes()

    return io.BytesIO(result_bytes)
