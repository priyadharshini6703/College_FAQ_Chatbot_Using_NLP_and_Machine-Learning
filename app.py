import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 20px;
}

.title {
    text-align: center;
    font-size: 35px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

data = {

    "question": [

        # ---------------- COURSES ----------------

        "What courses do you offer?",
        "Which courses are available?",
        "What programs are offered?",
        "What degrees are available?",
        "Tell me about the courses",
        "What programs can I study?",
        "What are the available courses?",

        # ---------------- FEES ----------------

        "What is the course fee?",
        "How much is the fee?",
        "What is the tuition fee?",
        "How much does the course cost?",
        "Tell me the fee details",
        "What are the fees?",
        "How much do I need to pay?",

        # ---------------- ADMISSION ----------------

        "When does admission start?",
        "When can I apply?",
        "What is the admission date?",
        "How can I apply for admission?",
        "Tell me about admission",
        "How do I get admission?",
        "What is the admission process?",

        # ---------------- CONTACT ----------------

        "How can I contact the college?",
        "What is the college contact number?",
        "How can I reach you?",
        "What is the phone number?",
        "Give me the contact details",
        "How do I contact the college?",
        "Where can I contact you?",

        # ---------------- LOCATION ----------------

        "Where is the college located?",
        "What is the college location?",
        "Where is your college?",
        "Tell me the college address",
        "How can I reach the college?",

        # ---------------- TIMINGS ----------------

        "What are the college timings?",
        "When does the college open?",
        "When does the college close?",
        "What is the working time?",
        "Tell me the college timings",

        # ---------------- PLACEMENT ----------------

        "Does the college provide placements?",
        "Is placement available?",
        "Tell me about placements",
        "Does the college have placement opportunities?",
        "What about campus placements?",

        # ---------------- HOSTEL ----------------

        "Is hostel available?",
        "Does the college provide hostel facilities?",
        "Is there a hostel?",
        "Tell me about hostel facilities",
        "Do you have hostel accommodation?"

    ],


    "intent": [

        # COURSES

        "courses",
        "courses",
        "courses",
        "courses",
        "courses",
        "courses",
        "courses",

        # FEES

        "fees",
        "fees",
        "fees",
        "fees",
        "fees",
        "fees",
        "fees",

        # ADMISSION

        "admission",
        "admission",
        "admission",
        "admission",
        "admission",
        "admission",
        "admission",

        # CONTACT

        "contact",
        "contact",
        "contact",
        "contact",
        "contact",
        "contact",
        "contact",

        # LOCATION

        "location",
        "location",
        "location",
        "location",
        "location",

        # TIMINGS

        "timings",
        "timings",
        "timings",
        "timings",
        "timings",

        # PLACEMENT

        "placement",
        "placement",
        "placement",
        "placement",
        "placement",

        # HOSTEL

        "hostel",
        "hostel",
        "hostel",
        "hostel",
        "hostel"
    ]
}


# Convert dictionary into DataFrame

df = pd.DataFrame(data)


# ============================================================
# TEXT PROCESSING
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)


X = vectorizer.fit_transform(df["question"])

y = df["intent"]


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)


# ============================================================
# CHATBOT RESPONSES
# ============================================================

responses = {

    "courses":
        "We offer various undergraduate and postgraduate programs. Please check the college course list for complete details.",

    "fees":
        "The course fee depends on the program. Please contact the college administration for the exact fee structure.",

    "admission":
        "Admissions are conducted according to the college admission schedule. Please contact the admission office for the current process.",

    "contact":
        "You can contact the college through the official phone number or email address.",

    "location":
        "The college is located at the institution's official campus address. Please check the official college website for the exact location.",

    "timings":
        "College working hours may vary. Please contact the college office for the current working timings.",

    "placement":
        "The college provides placement opportunities for eligible students. Please contact the placement cell for detailed information.",

    "hostel":
        "Hostel facilities may be available for students. Please contact the college administration for hostel availability and fees."
}


# ============================================================
# CHATBOT FUNCTION
# ============================================================
def chatbot(user_question):

    question_vector = vectorizer.transform([user_question])

    predicted_intent = model.predict(question_vector)[0]

    probabilities = model.predict_proba(question_vector)

    confidence = max(probabilities[0])

    print("Question:", user_question)
    print("Predicted intent:", predicted_intent)
    print("Confidence:", confidence)

    return responses[predicted_intent]


    # --------------------------------------------------------
    # UNKNOWN QUESTION HANDLING
    # --------------------------------------------------------

    if confidence < 0.35:

        return (
            "I'm sorry, I don't understand that question yet. "
            "Please ask about courses, fees, admission, contact, "
            "location, timings, placements, or hostel facilities."
        )


    # Return appropriate response

    return responses[predicted_intent]


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🤖 College FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about courses, admission, fees and college facilities.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Available Topics")

    st.write("You can ask about:")

    st.write("🎓 Courses")

    st.write("💰 Fees")

    st.write("📝 Admission")

    st.write("📞 Contact")

    st.write("📍 Location")

    st.write("🕐 Timings")

    st.write("💼 Placements")

    st.write("🏠 Hostel")


    st.divider()

    st.write("**Technology Used**")

    st.write("Python")

    st.write("TF-IDF")

    st.write("Scikit-learn")

    st.write("Logistic Regression")

    st.write("Streamlit")


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content":
                "Hello! 👋 I'm the College FAQ Chatbot. "
                "How can I help you?"
        }

    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_question = st.chat_input(
    "Type your question here..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_question:

    # Display user question

    with st.chat_message("user"):

        st.write(user_question)


    # Save user message

    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_question
        }

    )


    # Generate chatbot response

    answer = chatbot(user_question)


    # Display chatbot response

    with st.chat_message("assistant"):

        st.write(answer)


    # Save chatbot response

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )