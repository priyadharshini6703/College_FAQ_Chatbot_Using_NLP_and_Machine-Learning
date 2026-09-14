# College FAQ Chatbot Using NLP and Machine Learning

## Project Overview

This project is a College FAQ Chatbot developed using Python, Natural Language Processing (NLP), Machine Learning, and Streamlit.

The chatbot is designed to answer frequently asked questions related to college courses, fees, admission, contact details, location, timings, placements, and hostel facilities.

The chatbot uses TF-IDF to convert user questions into numerical features and Logistic Regression to classify the question into the appropriate category.

## Features

- Answers common college-related questions
- Uses Natural Language Processing
- Classifies questions into different categories
- Provides responses based on the predicted category
- Interactive web interface using Streamlit
- Maintains chat history during the session

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## FAQ Categories

The chatbot currently supports the following topics:

- Courses
- Fees
- Admission
- Contact
- Location
- Timings
- Placements
- Hostel

## Machine Learning Approach

### 1. Data Preparation

A dataset containing sample questions and their corresponding categories is created.

### 2. Text Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text questions into numerical features.

### 3. Classification

Logistic Regression is used to classify the user's question into the appropriate FAQ category.

### 4. Response Generation

After predicting the category, the chatbot provides the corresponding response.

## Project Workflow

User Question  
↓  
Text Preprocessing  
↓  
TF-IDF Vectorization  
↓  
Logistic Regression  
↓  
Intent/Category Prediction  
↓  
Relevant Answer  
↓  
Streamlit Chat Interface

## Example Questions

**User:** What courses do you offer?

**Chatbot:** We offer various undergraduate and postgraduate programs. Please check the college course list for complete details.

**User:** Is placement available?

**Chatbot:** The college provides placement opportunities for eligible students. Please contact the placement cell for detailed information.

**User:** Is hostel available?

**Chatbot:** Hostel facilities may be available for students. Please contact the college administration for hostel availability and fees.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-link>
