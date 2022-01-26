from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>SMS Spam Detector!</h1>' + '<p>testing second commit test</p>'

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)