from flask import Flask, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folders automatically
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)


# Low-pass filter function
def lowpass_filter(data, cutoff=0.1):

    b, a = butter(4, cutoff, btype='low')

    filtered = filtfilt(b, a, data)

    return filtered


# Home Page
@app.route('/')
def home():

    return '''
<!DOCTYPE html>

<html>

<head>

<title>Cloud Signal Analysis Platform</title>

<style>

body{
    font-family: Arial;
    background-color: #f4f4f4;
    text-align:center;
    padding-top:100px;
}

.container{
    background:white;
    width:500px;
    margin:auto;
    padding:40px;
    border-radius:10px;
    box-shadow:0px 0px 10px gray;
}

h1{
    color:#333;
}

button{
    background:#007BFF;
    color:white;
    border:none;
    padding:10px 20px;
    font-size:16px;
    border-radius:5px;
    cursor:pointer;
}

button:hover{
    background:#0056b3;
}

</style>

</head>

<body>

<div class="container">

<h1>Cloud Signal Analysis Platform</h1>

<p>Upload signal files for FFT analysis</p>

<form action="/analyze" method="post" enctype="multipart/form-data">

<input type="file" name="signalfile">

<br><br>

<button type="submit">Analyze Signal</button>

</form>

</div>

</body>

</html>
'''


# Analyze Signal
@app.route('/analyze', methods=['POST'])
def analyze():

    file = request.files['signalfile']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    # Load signal
    signal = np.loadtxt(filepath)

    # Add artificial noise
    noise = np.random.normal(0, 0.3, len(signal))

    noisy_signal = signal + noise

    # Apply filter
    filtered_signal = lowpass_filter(noisy_signal)

    # FFT
    fft_result = np.fft.fft(filtered_signal)

    # Create plots
    plt.figure(figsize=(10,10))

    # Original Signal
    plt.subplot(4,1,1)
    plt.plot(signal)
    plt.title("Original Signal")

    # Noisy Signal
    plt.subplot(4,1,2)
    plt.plot(noisy_signal)
    plt.title("Noisy Signal")

    # Filtered Signal
    plt.subplot(4,1,3)
    plt.plot(filtered_signal)
    plt.title("Filtered Signal")

    # FFT Analysis
    plt.subplot(4,1,4)
    plt.plot(np.abs(fft_result))
    plt.title("FFT Analysis")

    plt.tight_layout()

    # Save graph
    graph_path = os.path.join("static", "result.png")

    plt.savefig(graph_path)

    return f'''
    <h2>Signal Analysis Result</h2>

    <img src="/static/result.png" width="700">

    <br><br>

    <a href="/">Analyze Another Signal</a>
    '''


# Run App
if __name__ == '__main__':
    app.run(debug=True)