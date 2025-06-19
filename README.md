# AI-Powered Symptom Checker Chatbot

An intelligent medical symptom checker that uses BioBERT to analyze symptoms and generate context-specific follow-up questions, similar to Ada Health.

## Features

- **BioBERT-based symptom analysis**: Uses pre-trained medical language models
- **Dynamic question generation**: Context-aware follow-up questions based on symptoms
- **REST API**: Flask-based API for easy integration
- **Medical knowledge base**: Comprehensive symptom-to-question mapping
- **Lightweight and fast**: Optimized for real-time responses

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── biobert_classifier.py
│   │   ├── symptom_analyzer.py
│   │   └── question_generator.py
│   ├── data/
│   │   ├── symptoms_database.json
│   │   └── training_data.json
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/
│       ├── __init__.py
│       ├── medical_processor.py
│       └── response_builder.py
├── training/
│   ├── train_model.py
│   └── data_preparation.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── app.py
└── config.py
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download BioBERT model**:
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'); model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')"
   ```

3. **Run the API**:
   ```bash
   python app.py
   ```

4. **Test the API**:
   ```bash
   curl -X POST http://localhost:5000/api/analyze-symptom \
     -H "Content-Type: application/json" \
     -d '{"symptom": "I have a headache"}'
   ```

## API Endpoints

### POST /api/analyze-symptom
Analyze a symptom and get follow-up questions.

**Request**:
```json
{
  "symptom": "I have a headache",
  "session_id": "optional_session_id"
}
```

**Response**:
```json
{
  "symptom_category": "headache",
  "confidence": 0.95,
  "follow_up_questions": [
    "How long have you had this headache?",
    "Is the pain on one side or both sides?",
    "Do you also feel nausea or sensitivity to light?"
  ],
  "severity": "moderate",
  "recommendations": [
    "Consider over-the-counter pain relievers",
    "Rest in a quiet, dark room"
  ]
}
```

### POST /api/chat
Interactive chat endpoint for multi-turn conversations.

**Request**:
```json
{
  "message": "I have chest pain",
  "session_id": "user_session_123",
  "conversation_history": []
}
```

## Model Architecture

1. **Symptom Classification**: BioBERT-based classifier to categorize symptoms
2. **Question Generation**: Rule-based + ML approach for dynamic questions
3. **Severity Assessment**: Risk stratification based on symptom patterns
4. **Recommendation Engine**: Evidence-based medical recommendations

## Training

To train the model with custom data:

```bash
cd training
python train_model.py --data_path ../app/data/training_data.json
```

## Integration

This backend can be easily integrated with:
- React Native apps
- Next.js applications
- MERN stack projects
- Any REST API client

## Medical Disclaimer

This tool is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns. 