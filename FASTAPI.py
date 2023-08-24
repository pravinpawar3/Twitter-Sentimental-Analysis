from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from wordcloud import STOPWORDS
from typing import Dict 
from pydantic import BaseModel
import pandas as pd
import config
import re
import io
import csv


app = FastAPI()

model = None 
tokenizer = None
    
origins = [
    "http://localhost",
    "http://localhost:3000",  # Assuming your React app is running on port 3000
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up MongoDB connection
client = MongoClient(config.MONGO_CONNECTION_URL)
db = client[config.DATABASE_NAME]

@app.post("/upload/{data_type}")
async def upload_file(data_type: str, file: UploadFile = File(...)):
    contents = await file.read()
    # Save file to MongoDB
    collection = db[data_type]
    #collection.insert_one({"data": contents.decode("utf-8")})
    collection.insert_one({"filename": file.filename, "data": contents})
    return {"filename": file.filename}

@app.post("/upload/{data_type}")
async def upload_file(data_type: str, file: UploadFile = File(...)):
    contents = await file.read()
    # Save file to MongoDB
    collection = db[data_type]
    #collection.insert_one({"data": contents.decode("utf-8")})
    collection.insert_one({"filename": file.filename, "data": contents})
    return {"filename": file.filename}

@app.post("/upload/{data_type}")
async def upload_file(data_type: str, file: UploadFile = File(...)):
    contents = await file.read()
    # Save file to MongoDB
    collection = db[data_type]
    #collection.insert_one({"data": contents.decode("utf-8")})
    collection.insert_one({"filename": file.filename, "data": contents})
    return {"filename": file.filename}


@app.get("/download/{data_type}/{filename}")
def download_file(data_type: str, filename: str):
    # Retrieve file contents from MongoDB
    collection = db[data_type]
    result = collection.find_one({"filename": filename})
    if result:
        file_contents = result["data"]
        # Set response headers for CSV download
        response = Response(content=file_contents)
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["Content-Type"] = "text/csv"
        return response
    else:
        return {"error": "File not found"}
    
@app.delete("/delete/{data_type}/{filename}")
async def delete_file(data_type: str, filename: str):
    try:
        # Delete document from MongoDB
        collection = db[data_type]
        result = collection.delete_one({"filename": filename})
        if result.deleted_count == 1:
            return {"message": f"File {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get("/files/{data_type}")
def get_all_files(data_type: str):
    # Retrieve all filenames from MongoDB collection
    collection = db[data_type]
    results = collection.find({}, {"filename": 1})

    # Extract filenames from results
    filenames = [result["filename"] for result in results]

    return {"filenames": filenames}

@app.get("/pdf")
def generate_pdf():
    # Create PDF using ReportLab
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Dashboard Analysis Report")
    # Add content to the PDF, like analysis results
    c.save()
    buffer.seek(0)
    return FileResponse(buffer, filename="dashboard_report.pdf")

@app.get("/fetchData/{data_type}/{filename}")
def get_chart_data(data_type: str, filename: str) -> Dict:
    collection = db[data_type]
    result = collection.find_one({"filename": filename})
    df = pd.read_csv(io.StringIO(result['data'].decode('utf-8')))
    df = df[['airline_sentiment','airline','text']]
    # Process data for pie chart
    sentiment_counts = df['airline_sentiment'].value_counts()
    pie_chart_data = {
        'values': sentiment_counts.tolist(),
        'labels': sentiment_counts.index.tolist()
    }

    # Process data for bar chart
    sentiment_pos = df[df['airline_sentiment']=='positive']
    sentiment_neu = df[df['airline_sentiment']=='neutral']
    sentiment_neg = df[df['airline_sentiment']=='negative']
    
    sentiment_pos_airline = sentiment_pos["airline"].value_counts()
    sentiment_neu_airline = sentiment_neu["airline"].value_counts()
    sentiment_neg_airline = sentiment_neg["airline"].value_counts()
    
    barChartData = {
            "positive":      {
                    "values": sentiment_pos_airline.tolist(),
                    "labels": sentiment_pos_airline.index.tolist(),
                },
            "neutral":      {
                    "values": sentiment_neu_airline.tolist(),
                    "labels": sentiment_neu_airline.index.tolist(),
                },
            "negative":      {
                    "values": sentiment_neg_airline.tolist(),
                    "labels": sentiment_neg_airline.index.tolist(),
                }
    }
    
    # Process data for word cloud
    word_cloud_data = {}
    for sentiment_category in df['airline_sentiment'].unique():
        words = ' '.join(df[df['airline_sentiment'] == sentiment_category]['text'])
        filtered_words = [word for word in words.split()
                          if not word.startswith('@') and not word.startswith('http://') and word.lower() not in STOPWORDS
                          ]
        word_frequency = Counter(filtered_words)
        data = []
        data.extend([
            {  "text": word,  "value":  freq} for word, freq in word_frequency.items()
        ])
        word_cloud_data[sentiment_category] = data 

    return {
        'pieChartData': pie_chart_data,
        'barChartData': barChartData,
        'wordCloudData': word_cloud_data
    }
    
class Request(BaseModel):
    data: str
    
@app.post("/predict")
def predict(request_data: Request):
    global model, tokenizer
    data = request_data.data
    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax().item()
    return {"predicted_class": predicted_class}


@app.post("/load")
def load_model(request_data: Request):
    global model, tokenizer
    directory = request_data.data
    # Load the pre-trained model and tokenizer from local files
    #directory = "/Users/pravinanandpawar/Documents/project/Sentimental Analysis/Twitter-Sentimental-Analysis-main/saved_model"
    model = RobertaForSequenceClassification.from_pretrained(directory, num_labels=3)
    tokenizer = RobertaTokenizer.from_pretrained(directory)
    return "Model Loaded!"

@app.post("/extractModel")
def extract_model(request_data: Request):
    directory = request_data.data
    if model is not None:
        model.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    
    return {"message": f"Model extracted at path: {directory}"}

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    return text

@app.get("/train/{filename}")
def trainModel(filename: str):
    global model, tokenizer, db

    # Connect to MongoDB and read the CSV data
    collection = db["data"]
    csv_document = collection.find_one({"filename": filename})
    df = pd.read_csv(io.StringIO(csv_document['data'].decode('utf-8')))
    
    # Extract texts and labels from the DataFrame
    train_texts = df['text'].apply(preprocess_text).tolist()
    train_labels, _ = pd.factorize(df['airline_sentiment'])

    # Load pre-trained RoBERTa model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_texts))
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize texts and create input tensors
    inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    labels = torch.tensor(train_labels).to(device)

    # Create DataLoader directly from tensors
    train_dataloader = DataLoader(dataset=TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels), batch_size=32, shuffle=True)

    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
@app.get("/test/{filename}")
def testModel(filename: str):
    global model, tokenizer, db

    # Connect to MongoDB and read the CSV data
    collection = db["data"]
    csv_document = collection.find_one({"filename": filename})
    if csv_document:
        df = pd.read_csv(io.StringIO(csv_document['data'].decode('utf-8')))
        input_texts = df["text"].apply(preprocess_text).tolist()
        if input_texts is None:
            return {"message": "CSV data not found in MongoDB"}
        # Perform predictions
        batch_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            batch_logits = model(**batch_inputs).logits
            predicted_classes = batch_logits.argmax(dim=1).tolist()

        sentiment_labels = {2: "negative", 0: "neutral", 1: "positive"}
        df['output'] = [sentiment_labels[class_idx] for class_idx in predicted_classes]
        
        # Convert predictions to CSV format
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Input Text", "Predicted Class"])
        for input_text, prediction in zip(input_texts, predicted_classes):
            csv_writer.writerow([input_text, prediction])
        csv_data = csv_buffer.getvalue()

        # Connect to MongoDB and save the CSV and PDF as documents
        collection = db["data"]

        csv_document = {
            "filename": "predictions.csv",
            "data": csv_data
        }
        collection.insert_many([csv_document])
        
        actual_labels = df['airline_sentiment']  # Assuming 'airline_sentiment' is the column with actual sentiment labels

        # Calculate predictions as sentiment labels
        predicted_sentiments = [sentiment_labels[class_idx] for class_idx in predicted_classes]

        # Calculate accuracy
        accuracy = accuracy_score(actual_labels, predicted_sentiments)

        # Calculate precision, recall, and F1-score
        precision = precision_score(actual_labels, predicted_sentiments, average='weighted')
        recall = recall_score(actual_labels, predicted_sentiments, average='weighted')
        f1 = f1_score(actual_labels, predicted_sentiments, average='weighted')

        # Calculate confusion matrix
        confusion = confusion_matrix(actual_labels, predicted_sentiments, labels=list(sentiment_labels.values()))

        return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion.tolist()
    }
    else:
        return {"message": "CSV data not found in MongoDB"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)