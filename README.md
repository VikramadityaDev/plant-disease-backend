# Plant Disease Detection Backend

This is a backend service for **Plant Disease Detection** powered by a trained deep learning model.  
It provides a REST API to predict plant diseases from input images.

---

## Features
- Flask/FastAPI-based API (see `main.py`)
- Pre-trained deep learning model for plant disease classification
- JSON output with predicted class and confidence
- Ready-to-deploy on **Koyeb** with minimal configuration

---

## Project Structure
 ```text
    .
    ├── main.py
    ├── requirements.txt
    ├── runtime.txt
    ├── Procfile
    ├── class_indices.json
 ```

---

## Local Development

1. Clone this repository:
   
   ```
   git clone https://github.com/VikramadityaDev/plant-disease-backend.git
   cd plant-disease-backend
   ```
3. Create and activate a virtual environment:
   
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```
5. Install dependencies:
   
   ```
   pip install -r requirements.txt
   ```
7. Run the app locally:
   
   ```
   python main.py
   ```
   By default, the server should run on `http://127.0.0.1:5000` (or as defined in `main.py`).

## Deploy on Koyeb

1. Push your project to GitHub.
2. [![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/auth/signin)
3. Create a New App:
   - Select GitHub as the deployment source.
   - Choose your repository (`plant-disease-backend`).
   - Koyeb will automatically detect the `Procfile`, `requirements.txt` and `runtime.txt`.
4. Configure build & run:
   - Do nothing

Deploy and get your public API URL (e.g., `https://plant-disease.koyeb.app`).

## Example API Request

```
curl -X POST https://<your-app>.koyeb.app/predict \
     -F "file=@leaf.jpg"
```
## Response:
```
{
  "class": "Tomato___Late_blight",
  "confidence": 0.97
  "treatment": remove infected leaf
}
```

## Notes

   - The model (`plant_disease_fp32.tflite`) will be automatically downloaded from Google Drive.
   - `class_indices.json` contains the mapping of class IDs to disease names.
   - For larger ML models, consider using Koyeb’s persistent storage or external object storage.

## License

This project is licensed under the MIT License.
