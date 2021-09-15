import os

MODEL_PATH = r'models\res10_300x300_ssd_iter_140000.caffemodel'
PROTOTXT_PATH = r'models\deploy.prototxt.txt'
BOT_TOKEN = os.environ.get('BOT_TOKEN', '0')
APP_NAME = os.environ.get('APP_NAME', '0')
PORT = int(os.environ.get('PORT', '5000'))