import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- TITLE ----------
st.title("📰 Social Media Fake News Detection")
st.write("Detect whether a news statement is Fake or Real")

# ---------- DATASET ----------
data = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Source": ["News", "Gov", "Unknown", "Blog", "Social", "Official"],
    "Content": [
        "Government announces new policy",
        "Scientists discover new vaccine",
        "You won lottery click here",
        "Celebrity shocking scandal",
        "Fake cure for disease",
        "Economic growth report released"
    ],
    "Length": [5, 5, 5, 3, 4, 4],
    "Label": ["Real", "Real", "Fake", "Fake", "Fake", "Real"]
}

df = pd.DataFrame(data)

# ---------- SHOW TABLE ----------
st.subheader("📋 Dataset")
st.dataframe(df)

# ---------- MODEL ----------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Content"])
y = [1 if label == "Real" else 0 for label in df["Label"]]

model = MultinomialNB()
model.fit(X, y)

# ---------- GRAPH ----------
st.subheader("📊 Data Distribution")

label_counts = df["Label"].value_counts()

fig, ax = plt.subplots()
ax.bar(label_counts.index, label_counts.values)
ax.set_xlabel("News Type")
ax.set_ylabel("Count")
ax.set_title("Real vs Fake News Distribution")

st.pyplot(fig)

# ---------- INPUT ----------
st.subheader("🔍 Enter News Text")
user_input = st.text_area("Type news here")

# ---------- BUTTON ----------
if st.button("Check News"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        test = vectorizer.transform([user_input])
        prediction = model.predict(test)[0]

        if prediction == 1:
            st.success("✅ This is REAL news")
        else:
            st.error("❌ This is FAKE news")

        prob = model.predict_proba(test)[0]

        st.write(f"Confidence (Real): {prob[1]*100:.2f}%")
        st.write(f"Confidence (Fake): {prob[0]*100:.2f}%")