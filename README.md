## 🦷 DentAI Hospital System

FastAPI-based web system for dental hospital workflow with integrated deep learning model for dental disease image classification.

This README combines full project documentation with the model’s README for `dental_classifier_balanced.pth`.


## 📦 Project Overview

- FastAPI backend serving HTML pages and JSON APIs for students, doctors, college, secretary, and patients
- SQLite database (`dental_project_DB.db`)
- AI module for dental disease classification (ResNet-50)
- Frontend templates (no framework) under `templates/` and static uploads in `static/uploads/`


## 📁 Repository Structure

```
DentAI-Hospital-System/
├── app.py                         # FastAPI app: routes, endpoints, AI inference,
│                                  # file uploads, analytics
├── model.py                       # ResNet-50 model + inference utility class
├── queries.py                     # Core DB helpers (cases, queue, assignments)
├── Student_Page_Queries.py        # Student-side DB helpers and edit flows
├── doctor_query.py                # Doctor dashboard data and approvals/referrals
├── college_queries.py             # College dashboard data endpoints
├── dental_project_DB.db           # SQLite database
├── dental_classifier_balanced.pth # Trained model weights (download if missing)
├── static/
│   └── uploads/                   # Saved images from user uploads
├── templates/
│   ├── home.html                  # Landing + appointment booking
│   ├── Ai.html                    # AI image classification UI + chat UI
│   ├── book.html                  # Secretary dashboard
│   ├── college.html               # College dashboard
│   ├── doctor_all.html            # Doctor dashboard
│   ├── login_*.html               # Login pages (student/doctor/college/patient/secretary)
│   ├── patient.html               # Patient page
│   └── student.html               # Student page
└── README.md                      # This file
```


## 🧠 Dental Disease Classification (Deep Learning)

Advanced deep learning model for classifying dental diseases from images using ResNet-50 with balanced training and regularization.

### 🎯 Supported Disease Classes

1. Calculus (tartar buildup)
2. Dental Caries (tooth decay)
3. Gingivitis (gum inflammation)
4. Hypodontia (congenitally missing teeth)
5. Mouth Ulcer (oral lesions)
6. Tooth Discoloration (abnormal coloring)

### 🚀 Model Performance

| Metric | Score |
|-------:|:------|
| Training Accuracy | 94.17% |
| Validation Accuracy | 90.83% |
| Balanced Accuracy | ~90% |
| Overfitting Gap | 3.34% |

Low gap indicates strong generalization.

### 📥 Model Download

Due to size limits, download the weights externally and place them in the project root:

- Download: https://drive.google.com/file/d/1lR0T5pDCh5xH8MBh5uzYZSJyLYSShrrq/view?usp=sharing
- Save as: `dental_classifier_balanced.pth` in project root


## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Optional: CUDA-capable GPU for faster inference (PyTorch with CUDA)

### Install Dependencies

```bash
pip install fastapi uvicorn jinja2 pillow numpy torch torchvision
```

Optional utilities for local exploration and plotting (not required to run the server):

```bash
pip install matplotlib seaborn scikit-learn tqdm
```


## ▶️ Running the App

From the project root:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open in your browser:

- Home: http://127.0.0.1:8000/
- AI page: http://127.0.0.1:8000/AI
- Student: http://127.0.0.1:8000/student
- Doctor: http://127.0.0.1:8000/doctor
- College: http://127.0.0.1:8000/college
- Secretary dashboard (book): http://127.0.0.1:8000/book


## 🖼️ AI Inference Usage (API)

`POST /api/v1/AI/classification` with multipart form-data field `image`.

Response example:

```json
{
  "success": true,
  "predict": "Gingivitis"
}
```

Uploads are saved under `static/uploads/` with unique filenames.

Front-end UI for classification is available at `/AI` (drag-and-drop image and Analyze).


## 🔮 AI Inference Usage (Python script)

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import RegularizedDentalClassifier

checkpoint = torch.load('dental_classifier_balanced.pth', map_location='cpu')
model = RegularizedDentalClassifier(num_classes=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open('path/to/dental_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

class_names = checkpoint['class_names']
print('Predicted:', class_names[predicted.item()])
```


## 🌐 Key Pages and Flows

- `home.html` (/) — Landing + Book Appointment modal
  - API: `POST /api/v1/home/case` (create case)
- `book.html` (/book) — Secretary dashboard
  - API: `GET /api/home/show/cases` (queue)
  - API: `POST /api/v1/home/edit/case` (edit case basic info)
- `Ai.html` (/AI) — Image classification UI
  - API: `POST /api/v1/AI/classification`
  - Chat demo UI present; server streaming endpoint is currently commented out
- `doctor_all.html` (/doctor) — Doctor dashboard
  - API: `GET /api/v1/doctor/data/{doctorID}`
  - API: `GET /api/v1/doctor/student/cases/{doctorID}`
  - API: `POST /api/v1/approve/case` (approve/refuse/referral + notes)
- `college.html` (/college) — College overview
  - API: `GET /api/v1/college/batchs` | `/doctor` | `/student` | `/departments` | `/rounds`
- Login pages `/student/login`, `/doctor/login`, `/college/login`, `/patient/login`, `/secretary/login`
  - JSON APIs under `/api/v1/*/login` and `/api/v1/*/register` where applicable


## 🔌 API Highlights

Student
- `POST /api/v1/student/login` — email/password
- `POST /api/v1/student/register` — attach email/password to existing student id
- `GET  /api/v1/student/{student_id}` — basic profile
- `POST /api/v1/student/patient` — assign case to student in a department
- `GET  /api/student/cases/table/{student_id}` — student case table

Doctor
- `GET  /api/v1/doctor/data/{doctorID}` — profile + manager
- `GET  /api/v1/doctor/student/cases/{doctorID}` — cases to review
- `POST /api/v1/approve/case` — update description/treatment/approval/notes; optional referral

College
- `GET  /api/v1/college/batchs` | `/doctor` | `/student` | `/departments` | `/rounds`

Secretary / Patient Queue
- `POST /api/v1/home/case` — create case (from landing/secretary)
- `POST /api/v1/home/edit/case` — edit case basic info
- `GET  /api/home/show/cases` — queue (unassigned cases)

Analytics (student)
- `GET  /api/home/analyize/cases/{student_id}` — gender counts
- `GET  /api/home/analyize/checked/{student_id}` — pending/rejected/approved counts
- `GET  /api/home/analyize/case_by_department/{student_id}` — case counts by department
- `GET  /api/home/analyize/Treatment/{student_id}` — frequent treatments


## 🏗️ Model Architecture

```
RegularizedDentalClassifier
├── Backbone: ResNet-50 (ImageNet)
└── Head: Dropout(0.5) → Linear(2048→256) → ReLU → BN → Dropout(0.35) → Linear(256→6)
```

Training notes (from original model README): balanced training, class weighting, regularization, LR scheduling, early stopping, mixed precision.


## ⚠️ Notes and Known Gaps

- AI chat streaming endpoint is commented out server-side; the chat UI in `Ai.html` will fail to connect unless you enable it.
- Credentials are checked in plaintext; do not deploy as-is without adding password hashing and proper sessions/tokens.
- Model is currently loaded per request in `app.py`; for production, initialize once at startup and reuse.


## 📝 License

MIT License (add `LICENSE` file if distributing).


## 🙏 Acknowledgments

- ResNet-50 (Microsoft Research)
- PyTorch team
- Contributors and dataset providers


## 📣 Citation (Optional)

```bibtex
@software{dental_classifier_2025,
  author = {Your Name},
  title = {Dental Disease Classification using Deep Learning},
  year = {2025},
  url = {https://github.com/yourusername/dental-classification}
}
```


