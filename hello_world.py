from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import zipfile

app = Flask(__name__,template_folder = 'template')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CROPPED_IMAGES_FOLDER'] = 'cropped_images'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_contours', methods=['POST'])
def detect_contours():
    image_file = request.files['image_file']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_images = []
    for i, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cropped_img = img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
        cv2.imwrite(os.path.join(app.config['CROPPED_IMAGES_FOLDER'], f'{i}.jpg'), cropped_img)

    zip_filename = 'cropped_images.zip'
    zip_file_path = os.path.join(app.config['CROPPED_IMAGES_FOLDER'], zip_filename)
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for i, cropped_image in enumerate(cropped_images):
            zip_file.write(os.path.join(app.config['CROPPED_IMAGES_FOLDER'], f'{i}.jpg'))

    return send_file(zip_file_path, mimetype='application/zip', as_attachment=True)
    #return send_file(zip_file_path, mimetype='application/zip', attachment_filename=zip_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
