import os
import requests
from requests_oauthlib import OAuth1
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from collections import Counter, defaultdict
import json
import numpy as np
import dateutil.parser
from datetime import datetime
from networkx.algorithms.centrality.betweenness import betweenness_centrality
import operator

np.random.seed(42)

def get_tweets_from_screen_name(screen_name, start_date, end_date):
    """
    Get tweets from a screen name between two dates

    Args:
        screen_name (str): the twitter screen name
        start_date (str): the start date
        end_date (str): the end date

    Returns:
        list: tweets
    """
    
    tweets = []
    start_date = dateutil.parser.parse(start_date)
    end_date = dateutil.parser.parse(end_date)

    for tweet in json.load(open(os.path.join("data", f"{screen_name}.json"))):
        date = dateutil.parser.parse(tweet["created_at"], ignoretz=True)
        if date >= start_date and date <= end_date:
            tweets.append(tweet)
    return tweets
            
            
def get_pairs_from_tweet(tweet):
    """
    List all pairs of users mentioned in a tweet (excluding the user who tweeted)
        - Account retweeted
        - Account mentioned
        - Account in reply to

    Args:
        tweet (dict): the json formatted tweet
        
    Returns:
        list<tuple>: list of pairs of users
    """
    accounts = []
    if tweet["in_reply_to_screen_name"] != None:
        accounts.append(tweet["in_reply_to_screen_name"])
    for account in tweet["entities"]["user_mentions"]:
        accounts.append(account["screen_name"])
    if tweet.get("retweeted_status") != None:
        accounts.append(tweet["retweeted_status"]["user"]["screen_name"])
        for account in tweet["retweeted_status"]["entities"]["user_mentions"]:
            accounts.append(account["screen_name"])
    pairs = []
    for pair in combinations(list(set(accounts)), 2):
        pairs.append(tuple(sorted(pair)))
    return pairs


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.
    
    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions

    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, iterations=1, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def filter_nodes_to_keep(partition, weights, top_nodes=2):
    
    nodes_to_keep = []
    
    # Get all nodes per cluster
    clusters = defaultdict(list)
    for node, cluster in partition.items():
        clusters[cluster].append(node)
    
    for cluster, nodes in clusters.items():
        node_top_weights = sorted({node: weights[node] for node in nodes}.items(), key=operator.itemgetter(1), reverse=True)[:top_nodes]
        nodes_to_keep += [node for node, weight in node_top_weights]
    return nodes_to_keep
        
    

def generate_network_from_pairs(pairs, threshold=3):
    """
    Generate a networkx graph from a list of pairs of users

    Args:
        pairs (list<tuple>): list of pairs of users

    Returns:
        networkx.Graph: the graph
    """
    # Create the graph
    G = nx.Graph(random_state=42)
    for pair, count in Counter(pairs).items():
        if count >= 3:
            G.add_edge(pair[0], pair[1], weight=count)

    # compute the Louvain partition and get the layout
    partition = community_louvain.best_partition(G)
    pos = community_layout(G, partition)
        
    return G, partition, pos

def draw_graph(twitter_screen_name, tag, G, partition, pos, top_nodes=2):
    
    nodes_to_keep = filter_nodes_to_keep(partition, dict(G.degree()), top_nodes=top_nodes)
    
    plt.figure(figsize=(20,10))
    
    # Get labels to keep
    labels = {}
    for node in G.nodes():
        if node in nodes_to_keep and node != twitter_screen_name:
            labels[node] = node
        else:
            labels[node] = ""
            
    # color the nodes according to their partition
    cmap = cm.get_cmap('Pastel2', max(partition.values()) + 1)
    nx.draw_networkx_edges(G, pos, alpha=0.5, style='dashed', width=0.4)
    nx.draw_networkx_labels(G, pos, font_size=10, labels=labels, font_color="black")
    for node, color in partition.items():
        nx.draw_networkx_nodes(G, pos, [node], node_size=dict(G.degree)[node]*100, node_color=[cmap.colors[color]])
        
    plt.savefig(f"./output/{twitter_screen_name}_{tag}.png", dpi=300)
    plt.close()
    
def get_duration(start_date, end_date):
    start_date = dateutil.parser.parse(start_date)
    end_date = dateutil.parser.parse(end_date)
    return (end_date - start_date).days
    
    
def process_step(twitter_screen_name, president_screen_name, tag, start_date, end_date):
    
    duration = get_duration(start_date, end_date)

    tweets = get_tweets_from_screen_name(twitter_screen_name, start_date, end_date)
    pairs = []
    for tweet in tweets:
        pairs += (get_pairs_from_tweet(tweet))
    
    G, partition, pos = generate_network_from_pairs(pairs)
    
    
    try:
        draw_graph(twitter_screen_name, tag, G, partition, pos)
    except:
        print("Error while drawing graph: ", f"./output/{twitter_screen_name}_{tag}.png")
    
    centralities = sorted(dict(G.degree()).items(), key=operator.itemgetter(1), reverse=True)
    if len(centralities) > 1:
        if centralities[0][0] != twitter_screen_name:
            max_centrality = centralities[0]
        else:
            max_centrality = centralities[1]
    else:
        max_centrality = (None, None)
            
            
    betweenness_centralities = sorted(dict(nx.betweenness_centrality(G)).items(), key=operator.itemgetter(1), reverse=True)
    if len(betweenness_centralities) > 1:
        if betweenness_centralities[0][0] != twitter_screen_name:
            max_betweenness_centrality = betweenness_centralities[0]
        else:
            max_betweenness_centrality = betweenness_centralities[1]
    else:
        max_betweenness_centrality = (None, None)
        
        
    return {
        "parti": twitter_screen_name,
        "president": president_screen_name,
        "choc": tag,
        "start_date": start_date,
        "end_date": end_date,
        "nb_tweets": len(tweets),
        "avg_nb_tweets_day": len(tweets)/duration,
        "network_size": len(G.nodes()),
        "nb_unique_edges": len(set(pairs)),
        "nb_edges": len(pairs),
        "nb_communities": len(set(partition.values())),
        "network_density": nx.density(G),
        "max_centrality_value": max_centrality[0],
        "max_centrality_node": max_centrality[1],
        "max_betweenness_centrality_value": max_betweenness_centrality[0],
        "max_betweenness_centrality_node": max_betweenness_centrality[1],
        "president_centrality_value": dict(G.degree()).get(president_screen_name, 0),
        "president_betweenness_centrality_value": nx.betweenness_centrality(G).get(president_screen_name, 0)
    }

def get_date(date):
    try:
        return date.strftime("%Y-%m-%d")
    except:
        return None
    
    
if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint
    
    df = pd.read_excel("input_data.xlsx")
    results = []
    for row in df.to_dict("records"):
        party = row["Parti"]
        president = row["President"]
        # First step: graph with everything
        results.append(process_step(party, president, "ALL_TIME", "2019-02-26", "2022-09-01"))
        
        # Second step: one graph per choc period
        chocs = {header : get_date(value) for header, value in row.items() if header != "Parti" and header != "President" and get_date(value) != None}
        chocs = sorted(chocs.items(), key=operator.itemgetter(1))
        to_process = []
        for i, choc in enumerate(chocs):
            if i < len(chocs) - 1:
                next_choc = chocs[i+1]
                to_process.append((f"{choc[0]}_{next_choc[0]}", choc[1], next_choc[1]))

        for item in to_process:
            results.append(process_step(party, president, item[0], item[1], item[2]))
            
    df = pd.DataFrame(results)
    df.to_excel("output_data.xlsx", index=False)
