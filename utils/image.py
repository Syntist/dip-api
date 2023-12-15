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

def walsh_hadamard_matrix(n):
    if n == 0:
        return np.array([[1]])

    H_n_minus_1 = walsh_hadamard_matrix(n-1)
    top = np.concatenate((H_n_minus_1, H_n_minus_1), axis=1)
    bottom = np.concatenate((H_n_minus_1, -H_n_minus_1), axis=1)

    return np.concatenate((top, bottom), axis=0)
