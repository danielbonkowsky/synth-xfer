import re, math
import matplotlib.pyplot as plt
import numpy as np

def parse_infolog_exactness(file_path):
    """
    Parses info.log to extract exactness percentages
    Returns a list: [top_exactness, iter_0_exactness, iter_1_exactness, ...]
    """
    exactness_scores = []
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
        # 1. Extract 'Top Solution' Exactness
        # Looks for the pattern: Top Solution | Exact 60.3200% |
        top_match = re.search(r"Top Solution\s*\|\s*Exact\s*(\d+\.\d+)%", content)
        if top_match:
            exactness_scores.append(float(top_match.group(1)))
        else:
            print(f"Warning: 'Top Solution' format not found in {file_path}")

        # 2. Extract Iteration Exactness
        # Looks for the "Result of Current Solution" block followed by "exact%: "
        # We use re.DOTALL so the dot (.) matches newlines, capturing the multi-line block.
        # Pattern:
        # Result of Current Solution: 
        # bw: 8, dist: 596.88, exact%: 60.3200
        iter_matches = re.findall(r"Result of Current Solution:.*?exact%:\s*(\d+\.\d+)", content, re.DOTALL)
        
        # Convert strings to floats and add to list
        for match in iter_matches:
            exactness_scores.append(float(match))
            
        return exactness_scores

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def parse_perflog_times(file_path):
    """
    Parses a performance log file to extract the time taken for each iteration.
    Returns a list of floats: [iter 0 time, iter 1 time, ...]
    """
    iter_times = []
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
        # Regex to match the pattern: "Iter <number> took <time>s"
        # \s+ matches one or more whitespace characters
        # (\d+\.\d+) captures the floating point time value
        pattern = r"Iter\s+\d+\s+took\s+(\d+\.\d+)s"
        
        matches = re.findall(pattern, content)
        
        # Convert strings to floats
        iter_times = [float(time) for time in matches]
        
        return iter_times

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

if __name__ == "__main__":
    # Define your file lists (Must be ordered in pairs you want to compare)
    # Example: [Pair1_A, Pair1_B, Pair2_A, Pair2_B]
    log_files = ['outputs/ashr/info.log', 'outputs/ashr-base/info.log']
    perf_files = ['outputs/ashr/perf.log', 'outputs/ashr-base/perf.log']

    # --- Configuration ---
    LINES_PER_PLOT = 2  # How many lines you want on one graph
    
    # 1. Setup the grid dimensions
    # We divide total files by lines_per_plot to find how many plots we need
    num_plots = math.ceil(len(log_files) / LINES_PER_PLOT)
    cols = 2  # Adjusted columns for better visibility
    rows = math.ceil(num_plots / cols)

    plt.style.use('bmh')
    
    # Create figure with dynamic height based on rows
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # Ensure axes is always a flat list (handles 1 subplot case vs multiple)
    if num_plots == 1:
        axes_flat = [axes] if cols == 1 and rows == 1 else np.array([axes]).flatten()
    else:
        axes_flat = axes.flatten()

    # 2. Iterate through all files
    for i, (log_f, perf_f) in enumerate(zip(log_files, perf_files)):
        
        # Determine which subplot this file belongs to
        # i // 2 means indices 0 and 1 go to plot 0, indices 2 and 3 go to plot 1
        plot_idx = i // LINES_PER_PLOT
        ax = axes_flat[plot_idx]
        
        # Parse data
        scores = parse_infolog_exactness(log_f)
        lengths = parse_perflog_times(perf_f)
        
        # Calculate cumulative times
        times = [0]
        for length in lengths:
            times.append(times[-1] + length)
        
        # --- Plotting Changes ---
        
        # Generate a dynamic label based on folder name (e.g., "ashr" or "ashr-base")
        # This assumes structure outputs/<folder_name>/info.log
        try:
            label_name = log_f.split('/')[-2] 
        except IndexError:
            label_name = log_f 

        # We REMOVED the hardcoded color='#1f77b4'. 
        # Matplotlib will now automatically cycle colors (Blue for line 1, Orange for line 2)
        ax.plot(times, scores, linewidth=2, label=label_name, alpha=0.8, marker='o', markersize=4)

        # Set generic titles/labels only once per subplot (or overwrite safely)
        if i % LINES_PER_PLOT == 0:
            ax.set_title(f"Comparison Group {plot_idx + 1}", fontsize=12)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Exactness (%)', fontsize=10)

        ax.legend(loc='best', frameon=True, fontsize='small')

    # 3. Hide unused subplots
    # This cleans up the grid if you have an odd number of groups
    for j in range(num_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()