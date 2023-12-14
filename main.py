from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from model.dip import *
from utils.image import image_reader, result_bytes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return JSONResponse({"message": "Status OK"})

@app.post("/process_fft")
async def process_fft(file: UploadFile):
    try:
        image = await image_reader(file)

        fft = perform_fft(image)

        return StreamingResponse(result_bytes(fft), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process_dct")
async def process_dct(file: UploadFile):
    try:
        image = await image_reader(file)

        dct = perform_dct(image)

        return StreamingResponse(result_bytes(dct), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_walsh")
async def process_walsh(file: UploadFile):
    try:
        image = await image_reader(file)

        dct = perform_walsh_transform(image)

        return StreamingResponse(result_bytes(dct), media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/process_laplacian")
async def process_laplacian(kernel_size: int, sigma: int, file: UploadFile):
    try:
        image = await image_reader(file)

        log = perform_laplacian_of_gaussian(image, kernel_size, sigma)

        print(log)

        return StreamingResponse(result_bytes(log), media_type="image/png")
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process_equalization")
async def process_histogram_equalization(file: UploadFile):
    try:
        image = await image_reader(file)

        eq = perform_histogram_equalization(image)

        return StreamingResponse(result_bytes(eq), media_type="image/png")
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))