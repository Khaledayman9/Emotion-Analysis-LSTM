# Emotion-Analysis-LSTM
An LSTM-based model designed for emotion analysis of text. Using Long Short-Term Memory (LSTM) networks, the model processes and classifies text data into various emotion categories.

# Dataset
The dataset[^1] is a collection of documents and its emotions, It helps greatly in NLP Classification tasks. List of documents with emotion flag, Dataset is split into train, test & validation for building the machine learning model. It contains the emotions (sadness, anger, joy, fear, surprise).

[^1]: [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

# Aim

The aim of an LSTM-based emotion analysis application is to automatically detect and classify emotions expressed in text data. Here are some key objectives and benefits of such an application:

  1. Emotion Detection: The primary goal is to identify and categorize emotions (such as joy, sadness, anger, etc.) from text. This can be useful in understanding the sentiment or emotional tone of user-generated content.
  
  2. Enhanced User Experience: By understanding user emotions, applications can tailor responses, provide better customer support, or offer personalized recommendations based on emotional context.
  
  3. Content Analysis: It helps in analyzing large volumes of text data, such as social media posts, reviews, or feedback, to gauge public sentiment or monitor brand perception.
  
  4. Mental Health Monitoring: Emotion analysis can be used to detect signs of mental health issues or distress in users' text, potentially providing valuable insights for mental health professionals.
  
  5. Improved Communication: In applications like chatbots or virtual assistants, recognizing and responding appropriately to emotions can make interactions more natural and empathetic.
  
  6. Market Research: Businesses can use emotion analysis to understand consumer feelings towards products, services, or marketing campaigns, helping in decision-making and strategy development.

# Steps
## Forming and Cleaning Data
- **Load Data:** Import text data from files and structure it into a DataFrame.
- **Clean Data:** Process the text to remove unnecessary characters, handle missing values, and normalize text for consistency.


## Statistics and Exploration:
- **Explore Data:** Analyze basic statistics and distributions of the text and labels.
- **Visualize Data:** Create visualizations to understand the data distribution and potential imbalances.

## Tokenization and Padding:
- **Initialize Tokenizer:** Convert text into sequences of integers using a Tokenizer.
- **Fit Tokenizer:** Fit the tokenizer on the training text data.
- **Text to Sequences:** Convert text data into sequences of integers.
- **Pad Sequences:** Ensure uniform input length by padding sequences to a fixed length.

## Label Encoding and Class Weight Calculation:
- **Label Encoding:** Convert categorical labels into numerical format using LabelEncoder.
- **Class Weight Calculation:** Compute class weights to address class imbalance and ensure the model gives appropriate attention to underrepresented classes.

## Creating the Model:
### Define Model Architecture:
  - Embedding Layer: Converts input sequences into dense vectors of a fixed size.
    - Parameters: input_dim=20000 (size of the vocabulary), output_dim=64 (dimension of embedding vectors), input_length=100 (length of input sequences).
  - LSTM Layers: Processes sequences and captures dependencies in the data.
    - First LSTM Layer: units=64, return_sequences=True (returns the full sequence).
    - Second LSTM Layer: units=32 (returns only the last output).
  - Dropout Layer: Applies dropout to reduce overfitting.
    - Parameter: rate=0.5 (50% dropout rate).
  - Dense Layers: Fully connected layers for classification.
    - Hidden Dense Layer: units=32, activation='relu'.
    - Output Dense Layer: units=len(label_encoder.classes_) (number of classes), activation='softmax'.
### Compile Model:
  - Loss Function: loss='sparse_categorical_crossentropy' (suitable for multi-class classification).
  - Optimizer: optimizer='adam' (adaptive learning rate optimizer).
  - Metrics: metrics=['accuracy'] (evaluates model performance in terms of accuracy).

## Training the Model
### Fit Model:
  - Training Data: Model is trained on the train_padded data and train_labels.
  - Validation Data: Model is validated using the val_padded data and val_labels.
  - Epochs: Number of epochs set to 10 (number of complete passes through the training dataset).
  - Batch Size: Set to 32 (number of samples per gradient update).
  - Verbose: Set to 2 (level of verbosity during training).
### Evaluate Model:
  - Test Data: Model's performance is assessed on test_padded and test_labels.
  - Metrics: Test accuracy and loss are calculated to evaluate the model's generalization performance.
  - At the last epoch, the training and validation accuracy and loss values were:
```accuracy: 0.9931 - loss: 0.0174 - val_accuracy: 0.9140 - val_loss: 0.3718```
# Results
## 

# Technologies
- Python
- TensorFlow
- Kaggle Notebook
