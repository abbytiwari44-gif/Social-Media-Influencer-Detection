import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# ---------- PAGE ----------
st.set_page_config(page_title="Influencer Detection", layout="centered")

st.title("📊 Social Media Influencer Detection")
st.write("Find most influential user using Degree Centrality")

# ---------- INPUT ----------
st.sidebar.header("Input")

nodes = st.sidebar.slider("Number of Users", 3, 10, 5)

# ---------- GRAPH CREATION ----------
G = nx.erdos_renyi_graph(nodes, 0.5)

# ---------- CENTRALITY ----------
centrality = nx.degree_centrality(G)

# Find top influencer
top_user = max(centrality, key=centrality.get)

# ---------- OUTPUT ----------
st.subheader("📊 Result")

st.write("Most Influential User:", top_user)

# ---------- GRAPH ----------
fig, ax = plt.subplots()

pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=800, ax=ax)

# Highlight influencer
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[top_user],
    node_size=1000,
    ax=ax
)

st.pyplot(fig)

# ---------- CONCLUSION ----------
st.subheader("📌 Conclusion")

st.write("""
- Network graph represents users
- Degree centrality finds most connected user
- Highest connections = Influencer
""")