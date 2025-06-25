from rdflib import Graph
from pyvis.network import Network
from google.colab import files

# Load the saved KG from TTL file
g = Graph()
g.parse("knowledge_graph.ttl", format="turtle")

# Create a network graph using pyvis
net = Network(notebook=True, height="600px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources="in_line")

# Track added nodes
nodes_added = set()

# Add triples as nodes and edges
for subj, pred, obj in g:
    subj_str = str(subj).split("/")[-1]
    pred_str = str(pred).split("/")[-1]
    obj_str = str(obj).split("/")[-1]

    # Add nodes only once
    if subj_str not in nodes_added:
        net.add_node(subj_str, label=subj_str)
        nodes_added.add(subj_str)
    if obj_str not in nodes_added:
        net.add_node(obj_str, label=obj_str)
        nodes_added.add(obj_str)

    # Add edge
    net.add_edge(subj_str, obj_str, label=pred_str)

# Save and download
net.save_graph("knowledge_graph.html")
files.download("knowledge_graph.html")
