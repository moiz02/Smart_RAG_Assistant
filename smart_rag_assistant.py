"""
Smart RAG Assistant
-------------------------
This script implements a Retrieval-Augmented Generation (RAG) assistant that:
- Stores user facts and builds a knowledge graph (KG) from them
- Stores facts in a vector database for semantic search
- Answers questions using both the KG and the vector database, with help from a language model
- Supports pruning (removing) unimportant or redundant information
- Can visualize the knowledge graph and pruning process

How to use:
- Run this script and follow the command-line prompts
- Add facts, ask questions, prune, visualize
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch, shutil, time, os, atexit, re
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rdflib import Graph, URIRef, Namespace
from huggingface_hub import login


# Authenticate for HuggingFace Hub 
login("HF_TOKEN")

# Load the Mistral-7B-Instruct model for text generation
MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float16)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, do_sample=False)

# Define a namespace for the knowledge graph
EX = Namespace("http://example.org/")

# Load the embedding model for vector search
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def normalize_pronouns(text, user="USER_NAME"):
    """
    Replace first-person pronouns in the input text with the user's name.
    This helps store facts from the user's perspective in a consistent way.
    """
    text = text.strip()
    text = re.sub(r'\bmy\b', f"{user}'s", text, flags=re.I)
    text = re.sub(r'\bi\b', user, text, flags=re.I)
    text = re.sub(r'\bme\b', user, text, flags=re.I)
    text = re.sub(r'\bmine\b', f"{user}'s", text, flags=re.I)
    return text

def extract_triple(text):
    """
    Extract a single (subject, predicate, object) triple from a fact string using robust regex patterns.
    Returns a tuple or None if no pattern matches.
    """
    text = text.lower().strip()
    
    # Comprehensive patterns for different fact types
    patterns = [
        # Basic "is" patterns
        (r"^(.+?)(?:'s name)?\s+is\s+(.+)$", "is"),
        (r"^(.+?)\s+are\s+(.+)$", "are"),
        
        # Like/dislike patterns
        (r"^(.+?)\s+(?:likes?|love|enjoy|enjoys)\s+(?!to\s)(.+)$", "likes"),
        (r"^(.+?)\s+(?:hates?|dislikes?|dont?\s+like|doesnt?\s+like)\s+(?!to\s)(.+)$", "dislikes"),
        
        # Possession patterns
        (r"^(.+?)\s+(?:has|have|owns?)\s+(?:a\s+)?(.+)$", "has"),
        (r"^(.+?)(?:'s)\s+(.+)$", "has"),  # e.g., "moiz's car"
        
        # Location patterns
        (r"^(.+?)\s+(?:lives?|resides?)\s+(?:in|at)\s+(.+)$", "lives_in"),
        (r"^(.+?)\s+(?:works?|studies?)\s+(?:in|at)\s+(.+)$", "works_at"),
        
        # Age patterns
        (r"^(.+?)(?:'s)?\s+age\s+is\s+(.+)$", "age_is"),
        (r"^(.+?)\s+is\s+(\d+)\s+years?\s+old$", "age_is"),
        (r"^(.+?)\s+is\s+(.+?)\s+years?\s+old$", "age_is"),
        
        # Preference patterns
        (r"^(.+?)\s+(?:prefers?|prefer)\s+(.+?)\s+over\s+(.+)$", "prefers"),
        (r"^(.+?)\s+(?:prefers?|prefer)\s+(.+)$", "prefers"),
        
        # Relationship patterns
        (r"^(.+?)(?:'s)?\s+friend(?:\s+name)?\s+is\s+(.+)$", "friend_is"),
        (r"^(.+?)\s+(?:and|is friends? with)\s+(.+?)(?:\s+are friends?)?$", "friend_of"),
        (r"^(.+?)\s+(?:is|are)\s+(?:the\s+)?(?:friend|friends?)\s+of\s+(.+)$", "friend_of"),
        
        # Action patterns (more specific)
        (r"^(.+?)\s+(?:likes? to|love to|enjoy to)\s+(.+)$", "likes_to"),
        (r"^(.+?)\s+(?:hates? to|dont? like to)\s+(.+)$", "dislikes_to"),
        (r"^(.+?)\s+(?:plays?|play)\s+(.+)$", "plays"),
        (r"^(.+?)\s+(?:studies?|study)\s+(.+)$", "studies"),
        (r"^(.+?)\s+(?:works?|work)\s+(?:as\s+)?(.+)$", "works_as"),
        
        # Description patterns
        (r"^(.+?)\s+(?:is|are)\s+(.+)$", "is"),
        (r"^(.+?)\s+(?:looks? like|resembles?)\s+(.+)$", "looks_like"),
        
        # Time patterns
        (r"^(.+?)\s+(?:was born|born)\s+(?:in|on)\s+(.+)$", "born_in"),
        (r"^(.+?)\s+(?:started?|began)\s+(.+?)\s+(?:in|on)\s+(.+)$", "started"),
    ]
    
    for pattern, predicate in patterns:
        match = re.match(pattern, text)
        if match:
            if len(match.groups()) == 2:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                # Clean up common issues
                subject = re.sub(r"^the\s+", "", subject)  # Remove leading "the"
                obj = re.sub(r"^the\s+", "", obj)  # Remove leading "the"
                return (subject, predicate, obj)
            elif len(match.groups()) == 3:
                subject = match.group(1).strip()
                obj1 = match.group(2).strip()
                obj2 = match.group(3).strip()
                subject = re.sub(r"^the\s+", "", subject)
                obj1 = re.sub(r"^the\s+", "", obj1)
                obj2 = re.sub(r"^the\s+", "", obj2)
                return (subject, predicate, f"{obj1} over {obj2}")
    
    return None

def extract_multiple_triples(text):
    """
    Extract multiple triples from facts that contain compound objects or multiple relationships.
    Returns a list of (subject, predicate, object) tuples.
    """
    text = text.lower().strip()
    triples = []
    
    # Handle "likes to" patterns first (before compound patterns)
    to_patterns = [
        (r"^(.+?)\s+(?:likes? to|love to|enjoy to)\s+(.+)$", "likes_to"),
        (r"^(.+?)\s+(?:hates? to|dont? like to)\s+(.+)$", "dislikes_to"),
    ]
    
    for pattern, predicate in to_patterns:
        match = re.match(pattern, text)
        if match:
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            # Clean up
            subject = re.sub(r"^the\s+", "", subject)
            triples.append((subject, predicate, obj))
            return triples
    
    # Handle compound objects with "and"
    compound_patterns = [
        # Likes with "and" (but not "likes to")
        (r"^(.+?)\s+(?:likes?|love|enjoy|enjoys)\s+(?!to\s)(.+?)\s+and\s+(.+)$", "likes"),
        # Dislikes with "and"
        (r"^(.+?)\s+(?:hates?|dislikes?|dont?\s+like|doesnt?\s+like)\s+(?!to\s)(.+?)\s+and\s+(.+)$", "dislikes"),
        # Has with "and"
        (r"^(.+?)\s+(?:has|have|owns?)\s+(.+?)\s+and\s+(.+)$", "has"),
        # Plays with "and"
        (r"^(.+?)\s+(?:plays?|play)\s+(.+?)\s+and\s+(.+)$", "plays"),
        # Studies with "and"
        (r"^(.+?)\s+(?:studies?|study)\s+(.+?)\s+and\s+(.+)$", "studies"),
    ]
    
    for pattern, predicate in compound_patterns:
        match = re.match(pattern, text)
        if match:
            subject = match.group(1).strip()
            obj1 = match.group(2).strip()
            obj2 = match.group(3).strip()
            # Clean up
            subject = re.sub(r"^the\s+", "", subject)
            obj1 = re.sub(r"^the\s+", "", obj1)
            obj2 = re.sub(r"^the\s+", "", obj2)
            triples.append((subject, predicate, obj1))
            triples.append((subject, predicate, obj2))
            return triples
    
    # Handle comma-separated lists
    comma_patterns = [
        (r"^(.+?)\s+(?:likes?|love|enjoy|enjoys)\s+(?!to\s)(.+?)(?:,\s*and\s+)?(.+)$", "likes"),
        (r"^(.+?)\s+(?:hates?|dislikes?|dont?\s+like|doesnt?\s+like)\s+(?!to\s)(.+?)(?:,\s*and\s+)?(.+)$", "dislikes"),
    ]
    
    for pattern, predicate in comma_patterns:
        match = re.match(pattern, text)
        if match:
            subject = match.group(1).strip()
            obj1 = match.group(2).strip()
            obj2 = match.group(3).strip()
            # Clean up
            subject = re.sub(r"^the\s+", "", subject)
            obj1 = re.sub(r"^the\s+", "", obj1)
            obj2 = re.sub(r"^the\s+", "", obj2)
            triples.append((subject, predicate, obj1))
            triples.append((subject, predicate, obj2))
            return triples
    
    # If no compound patterns match, try single triple extraction
    single_triple = extract_triple(text)
    if single_triple:
        triples.append(single_triple)
    
    return triples

def get_node_name(node):
    """
    Extract the name part from an RDF node (for display).
    """
    if hasattr(node, 'toPython'):
        return str(node.toPython()).split('/')[-1]
    return str(node).split('/')[-1]

# Main Assistant Class 
class SmartRAGAssistant:
    """
    The main class for the Smart RAG Assistant.
    Handles fact storage, knowledge graph, vector store, pruning, and Q&A.
    """
    def __init__(self, persist_path=None, kg_path="knowledge_graph.ttl", user_name="moiz"):
        """
        Initialize the assistant.
        persist_path: Directory for persistent vector store (default: ./memory_store)
        kg_path: File path for persistent knowledge graph (TTL format)
        user_name: Name to use for replacing first-person pronouns
        """
        self.persist_path = persist_path or "./memory_store"
        self.kg_path = kg_path
        self.user_name = user_name
        self.embedding_model = embedder
        self.text_memory = []  # List of all facts as strings
        self.db = None         # Chroma vector store
        self.kg = Graph()      # RDFLib knowledge graph
        self.generator = gen  # Language model pipeline
        self.node_importance = defaultdict(float)  # Node importance scores
        self.pruned_nodes = set()                  # Set of pruned node names
        self.pruning_history = []                  # List of pruning actions
        self.load_kg()  # Load KG from disk if available
        # Load vector store if it exists
        if os.path.isdir(self.persist_path) and os.listdir(self.persist_path):
            try:
                self.db = Chroma(persist_directory=self.persist_path, embedding_function=self.embedding_model)
                docs = self.db.get()
                # Restore text memory from stored documents
                self.text_memory = list(docs["documents"]) if isinstance(docs, dict) and "documents" in docs else []
                print(f"Loaded vector store from {self.persist_path} with {len(self.text_memory)} facts.")
            except Exception as e:
                print("Failed to load existing vector store, starting fresh:", str(e))
                self.text_memory, self.db = [], None

    def add_fact(self, fact):
     
        # Normalize pronouns for consistency
        fact = normalize_pronouns(fact.strip(), self.user_name)
        self.text_memory.append(fact)
        # Add to vector store
        try:
            if self.db is None:
                self.db = Chroma.from_texts([fact], embedding=self.embedding_model, persist_directory=self.persist_path)
            else:
                self.db.add_texts([fact])
            print("Added to memory:", fact)
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["readonly", "permission", "write"]):
                print("Warning: Vector database is read-only. Facts will be stored in memory only.")
                print("To fix this, ensure the './memory_store' directory has write permissions.")
            else:
                print("Error saving to vector DB:", str(e))
        # Add to knowledge graph using robust regex patterns
        self._add_fact_to_kg(fact)
        # Update node importance
        self._update_node_importance()

    def _add_fact_to_kg(self, fact):
        
        triples = extract_multiple_triples(fact)
        if not triples:
            print(f"Could not extract any triples from: {fact}")
            return
            
        for subject, predicate, obj in triples:
            subj_clean = subject.strip()
            pred_clean = predicate.strip()
            obj_clean = obj.strip()
            # Replace spaces with underscores for URIs
            subj_uri = subj_clean.replace(" ", "_")
            pred_uri = pred_clean.replace(" ", "_")
            obj_uri = obj_clean.replace(" ", "_")
            self.kg.add((URIRef(EX[subj_uri]), URIRef(EX[pred_uri]), URIRef(EX[obj_uri])))
            print(f"Added triple: ({subj_clean}, {pred_clean}, {obj_clean})")

    def _update_node_importance(self):
       
        self.node_importance.clear()
        node_counts = Counter()
        for s, p, o in self.kg:
            node_counts[get_node_name(s)] += 1
            node_counts[get_node_name(p)] += 1
            node_counts[get_node_name(o)] += 1
        for node, count in node_counts.items():
            importance = count * 0.5
            # Bonus for being a subject
            importance += sum(1 for s, _, _ in self.kg if get_node_name(s) == node) * 0.3
            # Bonus for being a predicate
            importance += sum(1 for _, p, _ in self.kg if get_node_name(p) == node) * 0.2
            self.node_importance[node] = importance

    def prune_low_importance_nodes(self, threshold=0.5):
        
        self._update_node_importance()
        nodes_to_prune = {n for n, imp in self.node_importance.items() if imp < threshold and n not in self.pruned_nodes}
        if not nodes_to_prune:
            print("No nodes to prune at current threshold.")
            return {"pruned_nodes": [], "remaining_nodes": len(self.node_importance)}
        # Remove triples containing pruned nodes
        triples_to_remove = [(s, p, o) for s, p, o in self.kg if any(get_node_name(x) in nodes_to_prune for x in [s, p, o])]
        for triple in triples_to_remove:
            self.kg.remove(triple)
        self.pruned_nodes.update(nodes_to_prune)
        pruning_info = {
            "timestamp": time.time(),
            "threshold": threshold,
            "pruned_nodes": list(nodes_to_prune),
            "pruned_triples": len(triples_to_remove),
            "remaining_triples": len(self.kg)
        }
        self.pruning_history.append(pruning_info)
        print(f"Pruned {len(nodes_to_prune)} nodes and {len(triples_to_remove)} triples")
        print(f"Remaining: {len(self.kg)} triples, {len(self.node_importance) - len(self.pruned_nodes)} nodes")
        return pruning_info

    def prune_similar_facts(self, similarity_threshold=0.8):
        
        if len(self.text_memory) < 2:
            print("Not enough facts to prune for similarity.")
            return {"pruned_facts": [], "remaining_facts": len(self.text_memory)}
        # Get embeddings for all facts
        embeddings = self.embedding_model.embed_documents(self.text_memory)
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        # Find pairs of facts that are too similar
        similar_pairs = [(i, j, similarity_matrix[i][j]) for i in range(len(similarity_matrix)) for j in range(i+1, len(similarity_matrix)) if similarity_matrix[i][j] > similarity_threshold]
        if not similar_pairs:
            print("No similar facts found to prune.")
            return {"pruned_facts": [], "remaining_facts": len(self.text_memory)}
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        facts_to_remove = set()
        for i, j, _ in similar_pairs:
            if i not in facts_to_remove and j not in facts_to_remove:
                # Keep the fact with higher importance (or the first one if equal)
                if self.node_importance.get(self.text_memory[i].split()[0], 0) >= self.node_importance.get(self.text_memory[j].split()[0], 0):
                    facts_to_remove.add(j)
                else:
                    facts_to_remove.add(i)
        pruned_facts = [self.text_memory[i] for i in sorted(facts_to_remove, reverse=True)]
        for i in sorted(facts_to_remove, reverse=True):
            del self.text_memory[i]
        # Rebuild vector store
        if self.db and self.text_memory:
            try:
                shutil.rmtree(self.persist_path, ignore_errors=True)
                self.db = Chroma.from_texts(self.text_memory, embedding=self.embedding_model, persist_directory=self.persist_path)
            except Exception as e:
                print(f"Warning: Could not rebuild vector store: {e}")
        print(f"Pruned {len(facts_to_remove)} similar facts")
        print(f"Remaining: {len(self.text_memory)} facts")
        return {"pruned_facts": pruned_facts, "remaining_facts": len(self.text_memory), "similarity_threshold": similarity_threshold}

    def visualize_pruning(self, save_path="pruning_visualization.png"):
        
        if not self.pruning_history:
            print("No pruning history to visualize.")
            return
        # Build a directed graph for visualization
        G = nx.DiGraph()
        for s, p, o in self.kg:
            G.add_node(get_node_name(s), node_type='subject')
            G.add_node(get_node_name(p), node_type='predicate')
            G.add_node(get_node_name(o), node_type='object')
            G.add_edge(get_node_name(s), get_node_name(o), predicate=get_node_name(p))
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        # Use a single color for all nodes for simplicity
        node_colors = 'skyblue'
        node_sizes = [max(100, self.node_importance.get(n, 0)*200) for n in list(G.nodes())]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        edge_labels = {(u, v): d['predicate'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        legend = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Node')
        ]
        plt.legend(handles=legend, loc='upper left')
        plt.title(f"Knowledge Graph Visualization\nNodes: {len(list(G.nodes()))}, Edges: {len(list(G.edges()))}\nPruned Nodes: {len(self.pruned_nodes)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()

    def get_pruning_stats(self):
       
        if not self.pruning_history:
            return {"message": "No pruning operations performed yet."}
        total_pruned_nodes = sum(len(pruning['pruned_nodes']) for pruning in self.pruning_history)
        total_pruned_triples = sum(pruning['pruned_triples'] for pruning in self.pruning_history)
        return {
            "total_pruning_operations": len(self.pruning_history),
            "total_pruned_nodes": total_pruned_nodes,
            "total_pruned_triples": total_pruned_triples,
            "current_nodes": len(self.node_importance) - len(self.pruned_nodes),
            "current_triples": len(self.kg),
            "pruning_history": self.pruning_history
        }

    def _get_generated_text(self, prompt):
     
        result = self.generator(prompt)
        if result is None:
            return ""
        result_list = list(result) if not isinstance(result, list) else result
        if not result_list:
            return ""
        first = result_list[0]
        if isinstance(first, dict) and 'generated_text' in first:
            return first['generated_text']
        return str(first)

    def ask_question(self, query, k=5):
       
        query = query.strip().lower()
        # Find relevant facts from KG and vector store
        facts = list(set(self._search_kg(query) + self._search_vector(query, k)))
        if not facts:
            return self._fallback_answer(query)
        # Format the prompt for the language model
        context = "\n".join(f"- {f}" for f in facts)
        prompt = f"[INST] Use the following facts to answer the question.\n{context}\n\nQuestion: {query}\nAnswer: [/INST]"
        resp = self._get_generated_text(prompt).split("[/INST]")[-1].strip()
        print("", resp)
        return resp

    def _search_kg(self, query):
       
        return [f"{get_node_name(s)} {get_node_name(p)} {get_node_name(o)}" for s, p, o in self.kg if any(w in f"{get_node_name(s)} {get_node_name(p)} {get_node_name(o)}" for w in query.split())]

    def _search_vector(self, query, k):
       
        if self.db and self.text_memory:
            results = self.db.similarity_search(query, k=min(k, len(self.text_memory)))
            return list({r.page_content for r in results})
        return []

    def _fallback_answer(self, query):
      
        prompt = f"[INST] Answer the following question:\n{query} [/INST]"
        resp = self._get_generated_text(prompt).split("[/INST]")[-1].strip()
        print("", resp)
        return resp

    def save_kg(self):
        
        try:
            self.kg.serialize(destination=self.kg_path, format="turtle")
            print(f"Knowledge Graph saved to {self.kg_path}")
        except Exception as e:
            print("Error saving KG:", str(e))

    def load_kg(self):
       
        if os.path.exists(self.kg_path):
            try:
                self.kg.parse(self.kg_path, format="turtle")
                print(f"Loaded KG from {self.kg_path}")
            except Exception as e:
                print("Failed to load KG:", e)

    def clear_memory(self):
       
        shutil.rmtree(self.persist_path, ignore_errors=True)
        self.text_memory, self.kg, self.db = [], Graph(), None
        self.node_importance.clear()
        self.pruned_nodes.clear()
        self.pruning_history.clear()
        print("Memory and knowledge graph cleared.")

# Main Command-Line Loop
if __name__ == "__main__":
    bot = SmartRAGAssistant()
    atexit.register(bot.save_kg)
    print("\nSmart RAG Assistant with Pruning Initialized!")
    print("Commands:")
    print("- 'add: your fact' - Store a fact")
    print("- 'ask: your question' - Ask a question")
    print("- 'prune: low_importance' - Prune low importance nodes")
    print("- 'prune: similar' - Prune similar facts")
    print("- 'visualize' - Show pruning visualization")
    print("- 'stats' - Show pruning statistics")
    print("- 'clear' - Clear memory")
    print("- 'exit' - Quit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            bot.save_kg()
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            bot.clear_memory()
        elif user_input.startswith("add:"):
            bot.add_fact(user_input[4:].strip())
        elif user_input.startswith("ask:"):
            bot.ask_question(user_input[4:].strip())
        elif user_input.startswith("prune: low_importance"):
            threshold = 0.5
            if "threshold:" in user_input:
                try:
                    threshold = float(user_input.split("threshold:")[1].strip())
                except:
                    pass
            bot.prune_low_importance_nodes(threshold)
        elif user_input.startswith("prune: similar"):
            threshold = 0.8
            if "threshold:" in user_input:
                try:
                    threshold = float(user_input.split("threshold:")[1].strip())
                except:
                    pass
            bot.prune_similar_facts(threshold)
        elif user_input.lower() == "visualize":
            bot.visualize_pruning()
        elif user_input.lower() == "stats":
            stats = bot.get_pruning_stats()
            print("Pruning Statistics:")
            for key, value in stats.items():
                if key != "pruning_history":
                    print(f"  {key}: {value}")
        else:
            print("Use 'add:', 'ask:', 'prune:', 'visualize', 'stats', 'clear', or 'exit'")
