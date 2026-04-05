import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- TITLE ----------
st.title("📰 Social Media Fake News Detection")
st.write("Enter news text and see prediction + graph")

# ---------- SAMPLE DATA ----------
texts = [
    "government announces new policy",
    "scientists discover vaccine",
    "win money click here now",
    "celebrity shocking scandal",
    "fake cure for disease",
    "economic growth report released"
]

labels = [1, 1, 0, 0, 0, 1]  # 1=Real, 0=Fake

# ---------- MODEL ----------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ---------- INPUT ----------
user_input = st.text_area("Enter News Text")

# ---------- BUTTON ----------
if st.button("Check News"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        test = vectorizer.transform([user_input])
        prediction = model.predict(test)[0]
        prob = model.predict_proba(test)[0]

        # ---------- RESULT ----------
        st.subheader("📊 Result")

        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.write(f"Confidence Real: {prob[1]*100:.2f}%")
        st.write(f"Confidence Fake: {prob[0]*100:.2f}%")

        # ---------- GRAPH ----------
        st.subheader("📈 Prediction Graph")

        labels_graph = ["Fake", "Real"]
        values = [prob[0]*100, prob[1]*100]

        fig, ax = plt.subplots()
        ax.bar(labels_graph, values)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Prediction Confidence")

        st.pyplot(fig)