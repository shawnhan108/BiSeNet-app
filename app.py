import io
import base64

from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn 
from torchvision import transforms

from config import model_src, PORT, DEBUG_MODE, max_width
from model import BiSeNet

app = Flask(__name__)

# define colours
part_colors = [[0, 0, 0], [0, 255, 0], [150, 30, 150], [255, 65, 255], [0, 80, 150], [65, 120, 170], 
                [210, 180, 220], [100, 100, 200], [125, 125, 255], [125, 175, 215], [125, 125, 125], 
                [0, 150, 255], [0, 255, 255], [255, 255, 0], [120, 225, 255], [255, 125, 125], [0, 0, 255],
                [255, 0, 0], [80, 150, 0]]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpeg', 'jpg']

to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NCLASSES = 19
model = BiSeNet(n_classes=NCLASSES)
model.load_state_dict(torch.load(model_src, map_location=torch.device(device)))
model.eval()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():

    # get file
    img_file = request.files['img']
    if img_file and allowed_file(img_file.filename):
        npimg = np.fromstring(img_file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # preprocess
        img = Image.fromarray(np.uint8(img))
        orig_w, orig_h = img.size
        img = img.resize((512, 512), Image.BILINEAR)
        orig = img.copy()
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        img_list = [img, orig]

        # inference
        with torch.no_grad():
            out = model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        # parse and colour
        im = np.array(orig)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing.copy().astype(np.uint8)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(0, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        result_img = Image.fromarray(cv2.cvtColor(vis_parsing_anno_color, cv2.COLOR_BGR2RGB)).convert('RGBA')

        new_w, new_h = max_width, int(orig_h / orig_w * max_width)
        result_img = result_img.resize((new_w, new_h))

        # output format
        output = io.BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        output = output.getvalue()
        output = base64.b64encode(output)

    return render_template('home.html', img=output.decode("utf-8"), orig_h=new_h, orig_w=new_w)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
