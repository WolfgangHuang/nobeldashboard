import time
import pickle
import plotly.graph_objects as go
import precompute_plots as pcp

# Example compute function


def load_figure_from_pickle(filename="pcp_tab_current.pkl"):
    with open(filename, 'rb') as f1:
        pcp_tab_current = pickle.load(f1)
    fig_bubbles_population = pcp_tab_current['fig_bubbles_population']
    return fig_bubbles_population




# Measure time for computation
def measure_compute_time():
    start = time.perf_counter()
    fig = pcp.generate_bubbles_population(pcp.df_nlpc_complete_population_log)
    #fig.show()
    end = time.perf_counter()
    return end - start

# Measure time for loading from pickle
# def measure_load_time(filename="figure.pkl"):
def measure_load_time():
    start = time.perf_counter()
    fig = load_figure_from_pickle("pcp_tab_geography.pkl")
    #fig.show()
    end = time.perf_counter()
    return end - start

# Compare times
load_times = [measure_load_time() for _ in range(5)]
compute_times = [measure_compute_time() for _ in range(5)]


# Print average times
print(f"Average compute time: {sum(compute_times) / len(compute_times):.4f} seconds")
print(f"Average load time: {sum(load_times) / len(load_times):.4f} seconds")
