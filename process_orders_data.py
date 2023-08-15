import warnings 
warnings.filterwarnings('ignore')

import os 
import sys 
import pickle
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm 
import networkx as nx 



def get_shortest_path_length(road_net_graph, orders_data, weight_var):
    ''' 
    Assumption:
        All road segments are 2-way BUT
        This isn't actually true for many roads! (see ./data/../original/map_orig/"segments_mode_connected.csv" ; "edges")
    '''
    shortest_path_lengths = [] # in metres or seconds

    print(f"Considering weighting variable : '{weight_var}'")
    for idx, request in tqdm(orders_data.iterrows(), total=len(orders_data)):
        start_node = request.rest_node 
        end_node = request.cust_node 
        sp_length = nx.shortest_path_length(road_net_graph, start_node, end_node, weight=weight_var) 
        shortest_path_lengths.append(sp_length) 
    
    return shortest_path_lengths


def check_connected(net_df):
    ''' 
    To check if a given graph (road_network) is connected.
    '''
    # take edges from the road_net
    edges = []
    for idx, entry in net_df.iterrows():
        edges.append((entry.u, entry.v))

    graph = nx.Graph()                      # Instantiate a graph
    graph.add_edges_from(edges)             # Add edges to the graph
    is_connected = nx.is_connected(graph)   # Check connectivity

    assert is_connected, "The road-network is NOT conected."
    if is_connected: print("The road-network is connected!")

    return is_connected


def process_and_generate_data(orders_data):
    print("Calculating required information (timestamps) ...")
    # [ Calculate and create new required columns ]
    ## get 'placed_ts'
    orders_data.loc[:, 'placed_ts'] = orders_data['placed_time'].copy()
    
    ## get 'prep_ts'
    orders_data = pd.merge(orders_data, rest_prep_time[['rest_id', 'prep_time']], on='rest_id')
    orders_data['prep_time'] = orders_data['prep_time'].apply(lambda x:int(x)) 
    orders_data['prep_ts'] = orders_data['placed_time'] + orders_data['prep_time']
    
    ## get 'deliver_ts' (also 'deliver_dist' and 'deliver_time') 
    road_net_graph = nx.Graph() 
    for idx, edge in tqdm(road_net.iterrows(), total=len(road_net)):
        road_net_graph.add_edge(edge.u, edge.v, dist=edge.dist, time=edge.time)

    deliver_dists = get_shortest_path_length(road_net_graph, orders_data, weight_var='dist')
    deliver_times = get_shortest_path_length(road_net_graph, orders_data, weight_var='time')
    deliver_dists, deliver_times = np.array(deliver_dists), np.array(deliver_times)

    num_requests = orders_data.shape[0]
    orders_data['deliver_dist'] = deliver_dists.reshape(num_requests, 1).astype(int)
    orders_data['deliver_time'] = deliver_times.reshape(num_requests, 1).astype(int) 
    orders_data['deliver_ts'] = orders_data['prep_ts'] + orders_data['deliver_time']
    orders_data['deliver_ts'] = orders_data['deliver_ts'].astype(int)

    return orders_data      


def process_and_generate_data_slotted(orders_data, timestep):
    ''' 
    timestep: in seconds
    All orders placed within timestep duration of time are given the same placed_ts
    !! After slotting, the prep_ts's might not be unique !!
    '''
    def perform_slotting(orders_data, timestep): 
        orders_data['placed_ts'] = orders_data['placed_ts'].apply(lambda x:(x//timestep))
        orders_data['prep_ts'] = orders_data['prep_ts'].apply(lambda x:(x//timestep))
        orders_data['deliver_ts'] = orders_data['deliver_ts'].apply(lambda x:(x//timestep)) 
        return orders_data

    orders_data = process_and_generate_data(orders_data)
    orders_data = perform_slotting(orders_data, timestep)
    return orders_data 


def convert_even_timestamps(orders_data):
    ## convert all timestamps with odd values to even values (so that the total number of unique timestamps get almost halved)
    orders_data['placed_ts'] = orders_data['placed_ts'].apply(lambda x:x+1 if x%2==0 else x)
    orders_data['prep_ts'] = orders_data['prep_ts'].apply(lambda x:x+1 if x%2==0 else x)
    orders_data['deliver_ts'] = orders_data['deliver_ts'].apply(lambda x:x+1 if x%2==0 else x)
    return orders_data


def ensure_distinct_prepared_timestamps(orders_data, window_size, step_size):
    ''' 
    step_size=1 => consider all timestamps 
    step_size=1 => consider only even/only odd timestamps (depending on the input prep_ts's parity)
    '''
    ## ensure that all orders have distinct prep timestamps (since 2 drivers should not move towards 2 different orders prepared at the same timestamp)
    ## since we only have time-precision upto a second in the original data, so we manipulate it a little bit for our purpose
    ## begin_ts<=>placed_ts; end_ts<=>prep_ts; deliver_ts<=>terminal_ts
    max_ts = np.max(orders_data.deliver_ts.values)
    possible_timestamps = set([ts for ts in range(int(max_ts)+1)])
    unique_end_timestamps = set(orders_data.prep_ts.unique()) 
    available_timestamps = possible_timestamps.difference(unique_end_timestamps)

    # take all duplicates except for the first occurences 
    duplicated_indices = orders_data.duplicated(subset=['prep_ts'], keep='first')
    
    if duplicated_indices.any():
        duplicated_rows = orders_data[duplicated_indices]
        num_duplicates = duplicated_rows.shape[0]
        print(f"Number of time two (or more) orders are placed at the same time : {num_duplicates}")
        # print(duplicated_rows.sort_values(by='prep_ts'))

        # This shifting for non-duplication might cause prep_ts to become less than placed_ts or more than deliver_ts, both of which are undesirable
        # Therefore, we shift the placed_ts and deliver_ts for the respective order with the same shift value as well
        if step_size==1:
            window = range(step_size, window_size+1) 
        elif step_size==2:
            window = range(step_size, window_size+1, 2)
        
        for idx, order in duplicated_rows.iterrows():
            curr_ts = order.prep_ts
            # Non-duplication strategy (heuristic): choose from the previous or next 'window_size' timestamps which are available
            for w_idx in window:
                # breakpoint()
                prev_ts = curr_ts - w_idx
                next_ts = curr_ts + w_idx
                if prev_ts in available_timestamps:
                    # order.prep_ts = prev_ts # WRONG
                    orders_data.loc[idx, 'prep_ts'] = prev_ts
                    orders_data.loc[idx, 'placed_ts'] -= w_idx
                    orders_data.loc[idx, 'deliver_ts'] -= w_idx
                    if prev_ts < 0:
                        orders_data.loc[idx, 'placed_ts'] = 0
                    available_timestamps.remove(prev_ts)
                    break 
                elif next_ts in available_timestamps:
                    # order.prep_ts = next_ts # WRONG
                    orders_data.loc[idx, 'prep_ts'] = next_ts
                    orders_data.loc[idx, 'placed_ts'] += w_idx
                    orders_data.loc[idx, 'deliver_ts'] += w_idx
                    available_timestamps.remove(next_ts)
                    break
            
            # print(orders_data[orders_data.duplicated(subset=['prep_ts'], keep='first')].shape[0])
            if w_idx==window_size:        
                # print(idx)
                breakpoint()
                assert False, "Both prev_ts and next_ts are not available, please use another non-duplication strategy."
    return orders_data


def ensure_distinct_end_terminal_timestamps(orders_data):
    ## with orders with zero delivery time, we add some fixed cost derived from the data
    ## this, implicitly, ensures non-zero deliver_dist (or deliver_time)
    
    ## ensure that deliver_ts > prep_ts for all requests/orders \
    ## since, in the flow-network we don't want vertical s_r-t_r edges i.e., both end_r and terminal_r having the same timestamp
    
    # fixed_last_mile_cost = np.mean(orders_data.deliver_time.values)
    deliver_times = orders_data.deliver_time.values
    non_zero_deliver_times = deliver_times[deliver_times>=0]
    fixed_last_mile_cost = np.min(non_zero_deliver_times)
    orders_data.loc[orders_data['deliver_ts'] <= orders_data['prep_ts'], 'deliver_ts'] += int(fixed_last_mile_cost)
    
    # deliver_ts == prep_ts could happen for cases with shortest path distance=0 b/w rest_node and cust_node 
    # deliver_ts < prep_ts could happen for the above mentioned cases if they're manipulated to get 
    return orders_data


def sanity_check(orders_data):
        ## verify that all the orders should be prepared at different timestamps 
        # <=> "end_nodes" (see flow_based_mip.py) must have distinct timestamps! 
        assert orders_data.shape[0]==np.unique(orders_data.prep_ts.values).shape[0],\
                                                'all orders MUST be placed at different timestamps'
        ## verify that prep_ts > placed_ts for all orders (it may happen due to application of 'ensure_distinct_prepared_ts')
        assert orders_data[orders_data.placed_ts > orders_data.prep_ts].shape[0]==0, \
                                'all orders must have order-prepared timestamp > order-placed timestamp'
        ## verify that deliver_ts > prep_ts for all orders 
        assert orders_data[orders_data.deliver_ts < orders_data.prep_ts].shape[0]==0, \
                                'all orders must have order-deliver timestamp > order-prepared timestamp'
        return   


def save_as_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved successfully as {filename}")
    return


def process_generate_check_save(orders_data, filename):
    orders_data = process_and_generate_data(orders_data)
    orders_data = ensure_distinct_prepared_timestamps(orders_data, window_size=shift_window_size, step_size=1)
    # shift_window_size upto 1 hour is reasonable as it can't move peak-hour orders too far away from the peak hour
    orders_data = ensure_distinct_end_terminal_timestamps(orders_data)
    sanity_check(orders_data)
    save_as_csv(orders_data, filename)
    return 


def process_generate_slot_check_save(orders_data, timestep, filename):
    orders_data = process_and_generate_data_slotted(orders_data, timestep)
    # orders_data = ensure_distinct_prepared_timestamps(orders_data, window_size=shift_window_size, step_size=1)
    # shift_window_size upto 1 hour is reasonable as it can't move peak-hour orders too far away from the peak hour
    orders_data = ensure_distinct_end_terminal_timestamps(orders_data)
    # sanity_check(orders_data)
    save_as_csv(orders_data, filename)
    return 


def process_generate_odd_check_save(orders_data, filename):
    wn_mul = 5 
    step_size = 2

    orders_data = process_and_generate_data(orders_data)
    orders_data = convert_even_timestamps(orders_data)
    orders_data = ensure_distinct_prepared_timestamps(orders_data, window_size=wn_mul*shift_window_size, step_size=step_size)
    # shift_window_size upto 1 hour is reasonable as it can't move peak-hour orders too far away from the peak hour
    orders_data = ensure_distinct_end_terminal_timestamps(orders_data)
    sanity_check(orders_data)
    save_as_csv(orders_data, filename)
    return 


def map_driver_init_nodes_and_save(loc_df, loc2node_df, filename):
    # map the driver lat-lng to the nearest co-orindate's node 
    pass


if __name__=='__main__':
    # Program inputs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['A', 'B', 'C'], default='A', 
                        type=str, required=True, help='City name')
    parser.add_argument('--day', choices=[1, 2, 3, 4, 5, 6], default=1, 
                        type=int, required=True, help='Day index')
    parser.add_argument('--shift_window_size', choices=range(360), default=60, # 60 minutes 
                        type=int, required=False, help='Number of minutes by which an orders can be shifted so that distinct prep_ts can be achieved')
    parser.add_argument('--timestep', choices=range(3600), default=1,
                        type=int, required=True, help='timestep (in seconds) for slotting')
    # make required=False to go with default arguments when arguments are not provided  
    args = parser.parse_args()

    city = args.city  
    day = args.day 
    shift_window_size = args.shift_window_size*60
    timestep = args.timestep


    # Datapaths
    data_path = '/home/daman/Desktop/k_server/code/data/'
    name2id = {'A':10, 'B':1, 'C':4}
    id2name = {v:k for k,v in name2id.items()}


    # Load data
    print("Loading data ...")
    orders_data = pd.read_csv(os.path.join(data_path, 
                                           f'{city}/orders/{day}/orders.csv'), sep=' ').rename(columns={'node_id':'cust_node', 'restaurant_id':'rest_id'})
    rest_prep_time = pd.read_csv(os.path.join(data_path, 
                                              f'{city}/orders/{day}/rest_prep_time.csv'), sep=' ').rename(columns={'restaurant_id':'rest_id','sec':'prep_time','sec_std':'fpt_std'})
    rest_prep_time_slotted = pd.read_csv(os.path.join(data_path, 
                                                      f'{city}/orders/{day}/rest_prep_time_slotted.csv'), sep=' ')
    rest_to_node = pd.read_csv(os.path.join(data_path, f'{city}/orders/rest_id_to_node.csv'), 
                               header=None, sep=' ', names=['rest_id', 'rest_node'])

    drivers_data = pd.read_csv(os.path.join(data_path, f'{city}/drivers/{day}/driver_coords.csv'))
    loc2node_data = pd.read_csv(os.path.join(data_path, f'{city}/drivers/node_coord_map_{city}.csv'))
    idx2node = pd.read_csv(os.path.join(data_path, f'{city}/map/index_to_node_id.csv'), header=None, names=['idx', 'node_id'])
    node2idx = pd.read_csv(os.path.join(data_path, f'{city}/map/node_id_to_index.csv'), header=None, names=['node_id', 'idx'])
    
    road_net = pd.read_csv(os.path.join(data_path, f'{city}/map/{day}/u_v_time_dist'), header=None, sep=' ', names=['u', 'v', 'time', 'dist'])
    
    # Ensure that the road_network is connected 
    assert check_connected(road_net)

    # Take relevant columns from orders_data
    final_orders_data = orders_data[['order_id', 'rest_id', 'rest_node', 'cust_node', 'placed_time']]

    ## Note that we want distinct 'prep_ts' or 'arrival times' (or, equivalently, distince "end_nodes' timestamps") 
    ## There is no need of distinct 'placed_ts' or distinct 'deliver_ts'

    # Process data and create required columns
    # filename = os.path.join(data_path, f'{city}/orders/{day}/final_orders_{day}.csv')
    filename = os.path.join(data_path, f'{city}/orders/{day}/final_orders.csv')
    process_generate_check_save(final_orders_data, filename)
    # breakpoint() 

    # # Process data and create required columns + Slotting
    # filename = os.path.join(data_path, f'{city}/orders/{day}/final_orders_slotted_{timestep}.csv')
    # process_generate_slot_check_save(final_orders_data, timestep=30, filename=filename) 
    # breakpoint()
    
    # # Process data and create required columns + Only odd timestamps
    # filename = os.path.join(data_path, f'{city}/orders/{day}/final_orders_odd_timestamps.csv')
    # process_generate_odd_check_save(final_orders_data, filename)
    # breakpoint()

    # # Process and create driver initial locations file for given day 
    # filename = os.path.join(data_path, f'{city}/drivers/driver_init_nodes.csv')
    # map_driver_init_nodes_and_save(drivers_data, loc2node_data, filename)
    








