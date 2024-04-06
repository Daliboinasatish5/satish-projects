

from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

app = Flask(__name__)

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image, number_of_colors):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

    return hex_colors

def identify_colors(image):
    number_of_colors = 5  # You can adjust this based on your preference
    colors = []

    colors = get_colors(image, number_of_colors)

    return colors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify_colors', methods=['POST'])
def identify_colors_route():
    if 'image' in request.files:
        uploaded_image = request.files['image']
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        colors = identify_colors(image)
        return render_template('result.html', colors=colors)
    else:
        return render_template('error.html', error="No image uploaded.")

if __name__ == '__main__':
    app.run(debug=True)