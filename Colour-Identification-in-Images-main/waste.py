from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

app = Flask(__name__)

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors, show_chart):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        return hex_colors
    else:
        return rgb_colors

def match_image_by_color(image, color, threshold=60, number_of_colors=10): 
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))
    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if diff < threshold:
            select_image = True
    return select_image

def show_selected_images(images, color, threshold, colors_to_match):
    selected_images = []
    for i in range(len(images)):
        selected = match_image_by_color(images[i], color, threshold, colors_to_match)
        if selected:
            selected_images.append(images[i])
    return selected_images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    color = request.form['color']
    threshold = int(request.form['threshold'])
    images = []

    IMAGE_DIRECTORY = 'Images/images'
    for file in os.listdir(IMAGE_DIRECTORY):
        if not file.startswith('.'):
            images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))

    colors_to_match = 7  # You can adjust this value based on the number of colors you want to match

    if color.upper() in ['GREEN', 'BLUE', 'RED', 'YELLOW', 'WHITE']:
        color_rgb = [int(x) for x in colors[color.upper()]]
        selected_images = show_selected_images(images, color_rgb, threshold, colors_to_match)
        return render_template('result.html', images=selected_images)
    else:
        return render_template('error.html', error="Invalid color specified.")

if __name__ == '__main__':
    app.run(debug=True)
