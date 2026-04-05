import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- TITLE ----------
st.title("🛒 Supermarket Queue Simulation")
st.write("Basic simulation of customer waiting system")

# ---------- INPUT ----------
customers = st.number_input("Number of Customers", 10, 200, 50)
cashiers = st.number_input("Number of Cashiers", 1, 10, 2)
arrival_rate = st.slider("Arrival Rate (customers per minute)", 1, 10, 5)
service_rate = st.slider("Service Rate (customers per minute)", 1, 10, 6)

# ---------- BUTTON ----------
if st.button("Run Simulation"):

    arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, customers))
    service_times = np.random.exponential(1/service_rate, customers)

    start_times = []
    end_times = []
    wait_times = []

    current_time = [0]*cashiers

    for i in range(customers):
        cashier = np.argmin(current_time)

        start = max(arrival_times[i], current_time[cashier])
        end = start + service_times[i]

        current_time[cashier] = end

        start_times.append(start)
        end_times.append(end)
        wait_times.append(start - arrival_times[i])

    # ---------- OUTPUT ----------
    st.subheader("📊 Results")

    avg_wait = np.mean(wait_times)
    max_wait = np.max(wait_times)

    st.write(f"Average Wait Time: {avg_wait:.2f}")
    st.write(f"Maximum Wait Time: {max_wait:.2f}")

    # ---------- GRAPH ----------
    st.subheader("📈 Waiting Time Graph")

    fig, ax = plt.subplots()
    ax.plot(wait_times)
    ax.set_xlabel("Customer Number")
    ax.set_ylabel("Waiting Time")

    st.pyplot(fig)