from flask import Flask, render_template, request,jsonify
from PIL import Image
import base64, io
from ultralytics import YOLO
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app, origins=['http://127.0.0.1:5500'],
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     headers=['Content-Type'])

#  Load YOLOv8 model
model = YOLO('yolov8n.pt')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect_objects():
    print('Received request')
    if request.method == 'POST':
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))

        # Model prediction
        results = model.predict(image)

        results_list = []
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
            for box in boxes:  # Iterate over boxes
                r = box.xyxy[0].astype(int)  # Get corner points as int
                class_id = int(box.cls[0])  # Get class ID
                class_name = model.names[class_id]  # Get class name using the class ID
                confidence = round(float(box.conf[0]),4)  # Get confidence probability
                result_dict = {
                    "Class": class_name,
                    "Box-Coordinates": r.tolist(),
                    "Confidence": confidence
                }
                results_list.append(result_dict)
        print('Returning response:', {"prediction": results_list})
        return jsonify({"prediction": results_list})
    else:
        return jsonify({"error": "Invalid request method"})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
