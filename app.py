from fastapi import FastAPI, File, UploadFile;
from fastapi.middleware.cors import CORSMiddleware;
import tomato;
import common

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/tomatoModel")
async def TomatoModel(
    file: UploadFile = File(...)
):
    image = common.read_file_as_image(await file.read())
    # object classification of object before pre
    preObj= common.classification(image)
    if preObj["status"]==True and len(preObj["ObjectList"])>0:
        return preObj
    else:
        payload = tomato.TomatoModel(image)
        return payload

@app.get("/", tags=["Root"])

async def read_root():
    return {"message": "Welcome to the API!"}

