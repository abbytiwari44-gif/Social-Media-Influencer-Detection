import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# ---------- TITLE ----------
st.title("📊 Social Media Influencer Detection")
st.write("Identify most influential users using network centrality")

# ---------- INPUT ----------
num_users = st.slider("Number of Users", 5, 20, 8)
connection_prob = st.slider("Connection Probability", 0.1, 1.0, 0.4)

# ---------- BUTTON ----------
if st.button("Generate Network"):

    # Create random graph
    G = nx.erdos_renyi_graph(num_users, connection_prob)

    # ---------- CENTRALITY ----------
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Combine score
    influence_score = {}

    for node in G.nodes():
        influence_score[node] = (
            degree_centrality[node] + betweenness_centrality[node]
        )

    # Sort ranking
    sorted_users = sorted(influence_score.items(), key=lambda x: x[1], reverse=True)

    # ---------- OUTPUT ----------
    st.subheader("🏆 Influencer Ranking")

    for rank, (user, score) in enumerate(sorted_users, start=1):
        st.write(f"{rank}. User {user} → Score: {score:.3f}")

    # ---------- GRAPH ----------
    st.subheader("📈 Network Graph")

    pos = nx.spring_layout(G)

    fig, ax = plt.subplots()

    nx.draw(G, pos, with_labels=True, node_size=800, ax=ax)

    st.pyplot(fig)