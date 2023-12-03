import time

from multiprocessing import Manager

from ocular_simulator.server import TCPServer
from ocular_simulator.server import ClientHandlerMP

from ocular_simulator.sliding_window import SlidingWindow
from ocular_simulator.sliding_window import SlidingWindowMP

from ocular_simulator.detector import DetectorMP
from ocular_simulator.explanation import ExplainerMP

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def run_simulator_mp(tcp_server, 
                    msg_size, 
                    msg_format, 
                    n_streams,
                    start_ts,
                    sliding_window,
                    window_size, 
                    slide_size, 
                    slide_number,
                    active_fcms, 
                    snoise_models,
                    snoise_samples,
                    soutlier_scorers, 
                    smodel_rmse,
                    causal_graph,                              
                    sorted_nodes,
                    target_node,
                    m_samples, 
                    max_fcm_number,
                    nslide_used = 1,
                    num_noise_samples = 1500,
                    error_threshold_change=0.1,
                    shapley_config=None,
                    attribute_mean_deviation=False,
                    n_jobs=-1,
                    outlier_scorer = 'default',
                    dist_type=None):
    """
    ip : string ip address e.g. '127.0.0.1'
    slide_size : int, slide size in seconds
    window_size : int, window size in seconds
    start_ts : datetime.datetime, marked when the server starts running
    """
    start = time.perf_counter()
    clients = list()
    sliders = list()
    detectors = list()
    explainers = list()

    queues = list()
    sw_queues = list()
    wd_queues = list()
    we_queues = list()
    outlier_queues = list()
    explainer_queues = list()
    est_time_queues = list()

    manager = Manager()

    ## initialize threads for each client
    for _ in range(n_streams):
        conn, client_addr = tcp_server.server.accept()
        queue = manager.Queue()
        queues.append(queue)
        client = ClientHandlerMP(conn, client_addr, queues[-1], 
                                 slide_size, start_ts, 
                                 msg_size, msg_format)
        client.start()
        clients.append(client)

        #used to ensure sliding window is updated        
        sw_queue = manager.Queue()
        sw_queue.put(sliding_window)
        sw_queues.append(sw_queue)
        
        # used to get the latest sliding_window for detection
        wd_queue = manager.Queue()
        wd_queues.append(wd_queue)

        # used to get the latest sliding_window for explanation
        we_queue = manager.Queue()
        we_queues.append(we_queue)
        
        slider = SlidingWindowMP(queues[-1], sw_queues[-1], start_ts, slide_size, 
                                 wd_queues[-1], we_queues[-1])
        slider.start()
        sliders.append(slider)

        ## handling outlier detectors
        outlier_queue = manager.Queue()
        outlier_queues.append(outlier_queue)
        detector = DetectorMP(wd_queues[-1], 
                              slide_size,
                              window_size,
                              outlier_queues[-1],
                              start_ts,
                              causal_graph.features)
        detector.start()
        detectors.append(detector)

        ## handling outlier explanation
        explainer_queue = manager.Queue()
        explainer_queues.append(explainer_queue)
        est_time_queue = manager.Queue()
        est_time_queues.append(est_time_queue)
        explainer = ExplainerMP(window_size, 
                        slide_size, 
                        slide_number,
                        outlier_queues[-1],  
                        we_queues[-1], 
                        explainer_queues[-1],
                        active_fcms, 
                        snoise_models,
                        snoise_samples,
                        soutlier_scorers, 
                        smodel_rmse,
                        causal_graph,                              
                        sorted_nodes,
                        target_node,
                        m_samples, 
                        max_fcm_number,
                        nslide_used,
                        num_noise_samples,
                        error_threshold_change,
                        shapley_config,
                        attribute_mean_deviation,
                        n_jobs,
                        est_time_queues[-1],
                        outlier_scorer,
                        dist_type)
        explainer.start()
        explainers.append(explainer)

        logging.info(f'Number of active client {len(clients)}')
        
    
    for i in range(n_streams):
        clients[i].join()
        sliders[i].join()
        detectors[i].join()
        explainers[i].join()
    end = time.perf_counter()
    logging.info("finished initializing client thread")
    return explainer_queues, end-start, est_time_queues

    
