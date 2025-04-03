from flask import Flask, render_template
import subprocess
import webbrowser

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to start the Air Canvas script
@app.route('/start-canvas')
def start_canvas():
    subprocess.Popen(['python', 'air_canvas.py'])  # Runs the OpenCV script
    return "Air Canvas Started!"

# Route to start the Motion Detection script
@app.route('/start-motion-detection')
def start_motion_detection():
    subprocess.Popen(['python', 'motion_detection.py'])  # Runs the motion detection script
    return "Motion Detection Started!"

# Route to open ML Student Performance project
@app.route('/start-ml-project')
def start_ml_project():
    webbrowser.open("https://students-performance-xcj9.onrender.com")  # Opens the hosted ML project
    return "Redirecting to ML Project..."

# Route to start the Face Detection script
@app.route('/start-face-detection')
def start_face_detection():
    subprocess.Popen(['python', 'face.py'])  # Runs the face detection script
    return "Face Detection Started!"

if __name__ == "__main__":
    app.run(debug=True)
