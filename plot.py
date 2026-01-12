import re
import sys

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


# --- Example Usage ---
if __name__ == "__main__":
    # You can change these filenames to process specific log files
    log_files = ['outputs/abds/info-abds.log', 'outputs/add/info-add.log', 'outputs/ashr/info-ashr.log']
    perf_files = ['outputs/abds/perf-abds.log', 'outputs/add/perf-add.log', 'outputs/ashr/perf-ashr.log']

    for log_file in log_files:
        scores = parse_infolog_exactness(log_file)
        if scores:
            print(f"{log_file}: {scores}")
    
    for perf_file in perf_files:
        times = parse_perflog_times(perf_file)
        if times:
            print(f"{perf_file}: {times}")