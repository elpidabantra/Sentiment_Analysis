from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

# Metrics
# The first argument to Summary('request_processing_seconds', ...) is the name of the metric: 'request_processing_seconds'.
# The second argument is the description: 'Time spent processing requests'.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing requests') 
EXAMPLE_COUNTER = Counter('example_requests_total', 'Total requests processed')
CURRENT_LOSS = Gauge('model_current_loss', 'Current loss of the model')

"""
Explanation of Metrics:

Counter: tracks cumulative events (e.g., total number of requests or predictions).
Summary: measures durations or latencies.
Gauge: represents real-time values (e.g., current loss during training).

"""

# Example function to simulate monitoring
# When @REQUEST_TIME.time() is applied to the process_request() function, it records the start time when the function is called, 
# and once the function finishes executing, it records the end time. The time difference is then recorded as a metric under the 
# name request_processing_seconds (see above in Summary)

@REQUEST_TIME.time()     
def process_request():
    time.sleep(random.random())
    EXAMPLE_COUNTER.inc()

def monitor_training(loss_value):
    CURRENT_LOSS.set(loss_value)

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8000)  # Metrics available at http://localhost:8000/
    print("Prometheus metrics server running at http://localhost:8000/")

    # Example loop (replace with your training loop integration)
    while True:
        process_request()
        monitor_training(random.uniform(0, 1))
        time.sleep(1)
