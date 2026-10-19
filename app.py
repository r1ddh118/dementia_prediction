import sys
import os
from pathlib import Path

# Add project root directory to python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.app import MODEL_DIR, app

if __name__ == '__main__':
    if not (MODEL_DIR / 'dementia_model.joblib').exists():
        print("Model not found. Please run train.py first to train and save the model.")
    else:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG') == '1',
        )
