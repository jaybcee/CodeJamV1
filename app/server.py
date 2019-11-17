import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import random
import requests
import base64

export_file_url = 'https://drive.google.com/uc?export=download&id=1F2gh7W_KJ3BaFpFm_A4aJyO4lATclvvj'
export_file_name = 'export.pkl'

classes = ['0', '45', '90', '135']
path = Path(__file__).parent

hosted_images = path / 'static' / 'images'

# Static images are used to simulate live data that would be obtained by cropping full frame
# Data is actually predicted by model and has not been trained with.
front0 = hosted_images / 'front0.jpg'
front90 = hosted_images / 'front90.jpg'

back45 = hosted_images / 'back45.jpg'
back135 = hosted_images / 'back135.jpg'

kitchen0 = hosted_images / 'kitchen0.jpg'
kitchen135 = hosted_images / 'kitchen135.jpg'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


#
@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/eval-front', methods=['GET', 'POST'])
def front(request):
    rand = bool(random.getrandbits(1))
    if (rand):
        use_file = front0
        name = 'front0'
    else:
        use_file = front90
        name = 'front90'

    pred_class = prediction_from_img_path(use_file)
    return JSONResponse(format_g_res(pred_class, name))


@app.route('/eval-back', methods=['GET', 'POST'])
def front(request):
    rand = bool(random.getrandbits(1))
    if (rand):
        use_file = back45
        name = 'back45'
    else:
        use_file = back135
        name = 'back135'

    pred_class = prediction_from_img_path(use_file)
    return JSONResponse(format_g_res(pred_class, name))


@app.route('/eval-kitchen', methods=['GET', 'POST'])
def front(request):
    rand = bool(random.getrandbits(1))
    if (rand):
        use_file = kitchen0
        name = 'kitchen0'
    else:
        use_file = kitchen135
        name = 'kitchen135'

    pred_class = prediction_from_img_path(use_file)
    return JSONResponse(format_g_res(pred_class, name))


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


@app.route('/live', methods=['GET', 'POST'])
def process_stuff(request):
    r = requests.get('https://18c47516.ngrok.io/getframe').content
    imgdata = base64.b64decode(r)
    name = 'liveCapture.jpg'
    name_to_pass = 'liveCapture'
    filename = hosted_images / name  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    pred_class = prediction_from_img_path(filename)

    return JSONResponse(format_g_res(pred_class, name_to_pass))


def prediction_from_img_path(img_path):
    pred_class, pred_idx, outputs = learn.predict(open_image(img_path))
    return pred_class


def get_url_img(file_name):
    return f"https://dockersafehouse.appspot.com/static/images/{file_name}.jpg"


def format_g_res(angle, fname):
    temp = {"payload": {
        "google": {
            "expectUserResponse": False,
            "richResponse": {
                "items": [
                    {
                        "simpleResponse": {
                            "textToSpeech" : f'The subject of the image appears to be approximately at {angle} degrees.'

                        }
                    },
                    {
                        "basicCard": {
                            "subtitle": f"It appears that the object in the frame is at an angle of {angle}",
                            "image": {
                                "width": 400,
                                "height": 400,
                                "url": f"https://codejamhidden.onrender.com/static/images/{fname}.jpg",
                                "accessibilityText": "Picture of a lock"
                            },
                            "imageDisplayOptions": "CROPPED"
                        }
                    },
                ]
            }
        }
    }
    }

    print(temp)
    return temp


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8080, log_level="info")
