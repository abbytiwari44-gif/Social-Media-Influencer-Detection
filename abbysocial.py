import streamlit as st
import matplotlib.pyplot as plt

# ---------- TITLE ----------
st.title("📊 Interactive Misinformation Spread Model")
st.write("Graph updates based on your input values")

# ---------- INPUT ----------
total_users = st.slider("Total Users", 50, 500, 100)
initial_believers = st.slider("Initial Believers", 1, 50, 10)

beta = st.slider("Spread Rate (β)", 0.1, 1.0, 0.3)
gamma = st.slider("Recovery Rate (γ)", 0.1, 1.0, 0.1)

time_steps = st.slider("Time Steps", 10, 100, 50)

# ---------- MODEL ----------
S = [total_users - initial_believers]
I = [initial_believers]
R = [0]

for t in range(time_steps):
    new_infected = beta * S[-1] * I[-1] / total_users
    new_recovered = gamma * I[-1]

    S.append(S[-1] - new_infected)
    I.append(I[-1] + new_infected - new_recovered)
    R.append(R[-1] + new_recovered)

# ---------- GRAPH ----------
st.subheader("📈 Live Graph")

fig, ax = plt.subplots()

ax.plot(S, label="Skeptics")
ax.plot(I, label="Believers")
ax.plot(R, label="Fact-checkers")

ax.set_xlabel("Time")
ax.set_ylabel("Users")
ax.legend()

st.pyplot(fig)

# ---------- RESULTS ----------
st.subheader("📊 Results")

st.write(f"Peak Believers: {int(max(I))}")
st.write(f"Final Fact-checkers: {int(R[-1])}")
st.write(f"Remaining Skeptics: {int(S[-1])}")