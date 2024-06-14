import warnings
warnings.filterwarnings("ignore")

import os
import sys
import copy
import math
import pickle
import random
import argparse
import datetime
import itertools 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import gurobipy as gb
from gurobipy import GRB

import networkx as nx
from shapely import geometry
from itertools import product

random.seed(1234567)


def dt(y,m,d):
    return datetime.date(y, m, d)


def date_split(orders_data):
    all_orders = []
    num_orders = 0
    for d in date_range.keys():
        orders = copy.deepcopy(orders_data)
        st, end = date_range[d]
        # print(st,end)
        orders['order_date'] = orders['order_date'].astype('datetime64[ns]').dt.date
        orders = orders[(orders.order_date>=st) & (orders.order_date<=end)]
        orders = orders.reset_index(drop=True)
        print(orders.shape[0], end=' ')
        num_orders += orders.shape[0]
        all_orders.append(orders)
    print(f"Total number of orders: {num_orders}")
    return all_orders


def random_loc_generator(c_locs):
    lats, longs = c_locs[0], c_locs[1]
    coords = [(x,y) for x,y in zip(lats, longs)]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(longs), max(longs)
    new_lat, new_lng = random.uniform(min_lat, max_lat), random.uniform(min_lng, max_lng) 
    return [new_lat, new_lng]


def loc_in_zone(loc, c_locs):
    lats, longs = c_locs[0], c_locs[1]
    coords = [(x,y) for x,y in zip(lats, longs)]
    polygon = geometry.MultiPoint(coords).convex_hull
    Point_X, Point_Y = loc[0], loc[1]
    point = geometry.Point(Point_X, Point_Y)
    return point.within(polygon)


def generate_locs(m, c_locs):
    '''
    generate 'm' locations that lie within a given zone
    '''
    new_locs = []
    num_generated = 0
    while num_generated < m:
        new_loc = random_loc_generator(c_locs)
        sanity_check = loc_in_zone(new_loc, c_locs)
        if sanity_check:
            num_generated += 1
            new_locs.append(new_loc)
    return new_locs


def plot_customer_ffc(customer_locs, ffc_locs):
    plt.figure()
    plt.scatter(customer_locs[0],customer_locs[1],color='red',label='customers')
    plt.scatter(ffc_locs[:,0],ffc_locs[:,1],color='black',label='ffc_locs')

    plt.legend()
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Customer and Warehouse locations')


def compute_distance(loc1, loc2):
    '''
    calculate euclidean distance b/w facilty and customer sites
    '''
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return np.sqrt(dx*dx + dy*dy)


def kmedian(m, n, c, k):
    '''
    Problem: Find the locations of the ffcs
    MIP model formulation 
    Select k-ffc locations from the given set of potential locations
    m: number of potential locations
    n: number of customers
    c: cost matrix
    k: required number of ffcs/warehouses
    '''
    model = gb.Model("k-median")
    dem, fac = list(range(n)), list(range(m)) # demand and facility
    
    # VARIABLES:
    assign = model.addVars(dem, fac)
    select = model.addVars(fac, vtype=GRB.BINARY)  
    # OBJECTIVE: 
    model.setObjective(gb.quicksum(c[i, j]*assign[i, j] for j in fac for i in dem), GRB.MINIMIZE)
    # CONSTRAINTS:
    model.addConstrs(gb.quicksum(assign[i, j] for j in fac)==1 for i in dem) # each customer 'i' is assigned to exactly one facility
    model.addConstr(gb.quicksum(select)==k) # exactly k facilities out of the total m potential locations to be selected
    model.addConstrs(assign[i, j] <= select[j] for j in fac for i in dem) # a customer should only be assigned to a selected facility
    model.update() 
    model.Params.Method = 3
    model.optimize()
    
    return model, assign, select


def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    earth_radius = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate the distance
    distance = earth_radius * c
    
    return distance # in km


def picklify(data_struct, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_struct, f)
    print(f"Saved successfully as {filename}")
    return


if __name__=='__main__':
    data_path = "/home/daman/Desktop/k_server/code/data"

    # File paths
    print("Reading File Paths...")
    customer_dir = os.path.join(data_path, "Q/olist/olist_customers_dataset.csv")
    geolocation_dir = os.path.join(data_path, "Q/olist/olist_geolocation_dataset.csv")
    seller_dir = os.path.join(data_path, "Q/olist/olist_sellers_dataset.csv")
    order_dir = os.path.join(data_path, "Q/olist/olist_orders_dataset.csv")
    order_item_dir = os.path.join(data_path, "Q/olist/olist_order_items_dataset.csv")
    print("File Paths obtained !")

    # Creating dataframes
    print("\nLoading Files...")
    customer_df = pd.read_csv(customer_dir)
    location_df = pd.read_csv(geolocation_dir)
    seller_df = pd.read_csv(seller_dir)
    order_df = pd.read_csv(order_dir)
    order_item_df = pd.read_csv(order_item_dir)
    print("Done!")

    # more work for location dataframe:
    state_or_city = 'state'
    city_dataset = 'RJ'           
    state_dataset = 'RJ'         
    equal_or_not = 1


    # GENERAL DATA PROCESSING
    print("\nGeneral Data Processing...")
    orders = order_df[['order_id', 'customer_id', 'order_purchase_timestamp']] 
    sellers = seller_df[seller_df.seller_state==state_dataset]
    orders_sellers = order_item_df[['order_id','seller_id']]
    sellers = pd.merge(sellers, orders_sellers)
    customers = customer_df[customer_df.customer_state==state_dataset].drop('customer_unique_id', axis=1)
    locations = location_df[location_df.geolocation_state==state_dataset]
    orders_customers = pd.merge(orders, order_item_df, on='order_id').drop_duplicates('order_id')
    orders_customers = orders_customers[['order_id','customer_id','order_purchase_timestamp']] # seller_id doesn't matter since source nodes would come from the list of warehouse locations
    orders_customers['order_purchase_timestamp'] = orders_customers['order_purchase_timestamp'].astype('datetime64[ns]').dt.date

    # getting the destination locations corresponding to the orders
    customer_locations = pd.merge(customers, locations, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
    orders_with_destination = pd.merge(orders_customers, customer_locations, on='customer_id')
    orders_with_destination = orders_with_destination.drop_duplicates('order_id')
    orders_with_destination = orders_with_destination.rename(columns={'order_purchase_timestamp':'order_date','geolocation_lat':'cust_lat','geolocation_lng':'cust_lng'})
    orders_with_destination = orders_with_destination[['order_id','customer_id','order_date','cust_lat','cust_lng']].sort_values('order_date').reset_index()
    orders_with_destination['cust_loc'] = orders_with_destination.apply(lambda row:tuple([row['cust_lat'],row['cust_lng']]), axis=1)
    print("Done!")

    # LOCATIONS DATA
    print("\nGenerating Locations...")
    """Facility Location to get the 'k' warehouse locations"""
    orders_with_destination = orders_with_destination[orders_with_destination['cust_lng']>-46] # removing 3 undesirable outliers!!
    cust_locs_df = orders_with_destination[['cust_lat','cust_lng']].drop_duplicates().reset_index() # notice the drop_duplicates()!
    customer_locs = np.transpose(cust_locs_df.values)
    cust_locs_df['cust_node'] = cust_locs_df.index + 1

    num_potential_locs = 100
    potential_locs = np.array(generate_locs(num_potential_locs, customer_locs))
    plot_customer_ffc(customer_locs, potential_locs)

    pffc_locs = np.array(potential_locs)
    cust_locs = np.array([[lat,lng] for lat,lng in zip(customer_locs[0],customer_locs[1])])

    # Key parameters of MIP model formulation
    num_facilities = pffc_locs.shape[0]
    num_customers = cust_locs.shape[0]
    cartesian_prod = list(product(range(num_customers), range(num_facilities))) # [num_customers x num_facilities]

    # Delivering / Shipping costs:
    delivery_cost = {(c, f): compute_distance(cust_locs[c], pffc_locs[f]) for c, f in cartesian_prod}
    cost_matrix = np.array(list(delivery_cost.values())).reshape((num_customers, num_facilities))
    final_clocs = orders_with_destination[['cust_lat','cust_lng']].drop_duplicates()
    num_ffc_center =  final_clocs.shape[0]//20
    # 5-percent nodes are warehouses (in the real-world food-delivery dataset, #restaurants was around 5% of the total number of nodes (naturally the number of restaurant nodes was even lesser due to potential overlaps)).
    # It is still an overestimation of the total number of warehouse in a city because typically the number of restaurants is higher than the number of warehouses.
    _, assignz, selected_ffcs = kmedian(num_facilities, num_customers, cost_matrix, num_ffc_center) 

    mask_ffcs = [abs(int(val.x)) for val in selected_ffcs.values()]
    mask_ffc = np.array([True if val==1 else False for val in mask_ffcs])
    final_ffc_locs = potential_locs[mask_ffc]
    ffc_locs_df = pd.DataFrame(final_ffc_locs, columns=['rest_lat','rest_lng']).drop_duplicates().reset_index()
    ffc_locs_df['rest_node'] = ffc_locs_df.index + cust_locs_df.shape[0] + 1
    NUM_RESTS = len(ffc_locs_df)
    assert sum(mask_ffcs)==num_ffc_center, "The number of selected ffcs < required_num_ffc"

    # find out which customers are assigned to which ffc: (we assume that no new customer nodes or warehouse nodes would be opened within the dataset duration under consideration)
    cust_ffc_map = {key:abs(int(val.x)) for key, val in assignz.items()}
    cust_ffc_map = {cust: ffc for (cust, ffc) in assignz.keys() if cust_ffc_map[(cust, ffc)]==1}

    old_indices = [idx for idx, val in enumerate(selected_ffcs.values()) if val.x==1]
    # old_indices = np.unique(np.array(list(cust_ffc_map.values()))) # indices of the selected warehouses in 'potential_locs' # This would assume that each facility is allocated to at least one customer, which is not a valid assumption!
    new_indices = list(range(num_ffc_center)) # indices of the warehouses in 'final_ffc_locs' (see above) and locs_df (see below)
    # assert len(old_indices)==len(new_indices), "There exists a facility which is not assigned to any customer!"
    idx_map = {oidx:nidx for oidx, nidx in zip(old_indices, new_indices)}
    assert pffc_locs[old_indices].all()==final_ffc_locs.all(), "wrong idx_map!"

    for i in cust_ffc_map.keys():
        old_val = cust_ffc_map[i]
        cust_ffc_map[i] = idx_map[old_val] + cust_locs_df.shape[0] # since the rest nodes will places at the end in the final locs_df 
    tmp_map = {k+1:v for k,v in cust_ffc_map.items()} # since all the cust_node/rest_node are 1-indexed
    cust_ffc_map = tmp_map

    plot_customer_ffc(cust_locs, final_ffc_locs)    


    # locs_df = pd.concat([ffc_locs_df.rename(columns={'rest_lat':'lat','rest_lng':'lng'}), 
    #                     cust_locs_df.rename(columns={'cust_lat':'lat','cust_lng':'lng'})], axis=0).reset_index(drop=True)
    locs_df = pd.concat([
                            cust_locs_df.rename(columns={'cust_lat':'lat','cust_lng':'lng','cust_node':'node'}),
                            ffc_locs_df.rename(columns={'rest_lat':'lat','rest_lng':'lng', 'rest_node':'node'}),
                        ],axis=0).reset_index(drop=True)
    # breakpoint()
    # locs_df = locs_df.drop_duplicates(['lat','lng']) # allowing mutliple nodes to correspond to the same location, assuming imprecise lat-lng locations
    locs_df['rest_node'] = locs_df['node'] # this is correct for index>=num_customers the rest_node corresponding to warehouses would be its own node value
    locs_df['rest_node'][:num_customers] = locs_df['node'][:num_customers].map(cust_ffc_map) # the rest_nodes corresponding to the customer locs (all nodes except the last NUM_REST nodes) will be corrected!
    locs_df['rest_node'] = locs_df['rest_node'].apply(lambda x:int(x))
    locs_df['loc'] = locs_df.apply(lambda row:tuple([row['lat'],row['lng']]),axis=1)
    # last NUM_RESTS nodes in locs_df correspond to the restaurant locations

    # SAVING NODES.CSV #
    locs_df_path = os.path.join(data_path, f"Q/map/_nodes.csv")
    locs_df.to_csv(locs_df_path, index=False)
    print(f"Number of nodes: {locs_df.shape[0]}")
    print(f"Generated and Stored locations ([Warehouse, Customers]) data at {locs_df_path}")

    # Warehouse Data
    NUM_TIMESTAMPS = 5000
    MAX_PREP_TIME = 10 # NUM_TIMESTAMPS//1000 # Prep time (Packaging time) is really low in quick commerce setting
    MIN_DELIVERY_TIME = 1
    MAX_DELIVERY_TIME = 10*MAX_PREP_TIME # typically packaging time is less than delivery time in quick commerce (unlike food delivery) https://blinkit.com/faq#:~:text=How%20does%20Blinkit%20deliver%20in,of%20the%20customer%20placing%20it.)

    # rests = locs_df[locs_df.index<NUM_RESTS]
    rests = ffc_locs_df
    rests['prep_time'] = random.choices(range(2,MAX_PREP_TIME+1),k=NUM_RESTS)

    # SAVING WAREHOUSE LOCATIONS #
    rests_path = os.path.join(data_path, 'Q/orders/_rests.csv')
    rests.to_csv(rests_path, index=False)
    print(f"Number of Warehouses: {rests.shape[0]}")
    print(f"Generated and Stored Warehouse Locations at {rests_path}")

    # breakpoint()
    
    # Shortest Paths computation
    NUM_NODES = locs_df.shape[0]
    apsp = {i:
            {j:
                {'dist':None, 'time':None}
                for j in range(1,NUM_NODES+1)
                }
            for i in range(1,NUM_NODES+1)
        }
    apsp['src'] = {j: {'dist':0, 'time':0} for j in range(1,NUM_NODES+1)}

    paths ={i:
            {j:
                {'dist':[i,j], 'time':[i,j]} # since complete graph!
                for j in range(1,NUM_NODES+1)
                }
            for i in range(1,NUM_NODES+1)
        }
    paths['src'] = {j: {'dist':['src',j], 'time':['src',j]} for j in range(1,NUM_NODES+1)}

    dists = []
    for i in range(1,NUM_NODES+1):
        slat, slng = locs_df.iloc[i-1].lat, locs_df.iloc[i-1].lng
        for j in range(i, NUM_NODES+1):
            dlat, dlng = locs_df.iloc[j-1].lat, locs_df.iloc[j-1].lng 
            apsp[i][j]['dist'] = haversine_distance(slat, slng, dlat, dlng)
            # breakpoint()
            if apsp[i][j]['dist']>MAX_DELIVERY_TIME: 
                apsp[i][j]['dist'] = apsp[i][j]['dist']//500 # since some customer locations are too far away; note that the delivery dist is just reduced and is still unbounded
            apsp[i][j]['dist'] = max(MIN_DELIVERY_TIME, apsp[i][j]['dist'])
            apsp[i][j]['time'] = apsp[i][j]['dist']//2 # min(apsp[i][j]['dist'], MAX_DELIVERY_TIME) # assuming a constant speed of speed of 10km/timestep
            if i>=num_customers or j>=num_customers: 
                apsp[i][j]['time'] = apsp[i][j]['time']//2 # assuming that the warehouses are more easily reachable due to familiarity; this assumption is required to ensure low number of failed orders
            apsp[j][i]['dist'] = apsp[i][j]['dist']
            apsp[j][i]['time'] = apsp[i][j]['time']
            dists.append(apsp[i][j]['dist'])

    apsp_path = os.path.join(data_path, "Q/map/apsp.pkl")
    paths_path = os.path.join(data_path, "Q/map/paths.pkl")
    picklify(apsp, apsp_path)
    picklify(paths, paths_path)
    print(f"Generated and Stored Shortest Path values at {apsp_path} and the corresponding Paths at {paths_path}")

    # breakpoint()

    # Months to 7 Days
    print("\nGenerating Orders Data...")
    dates = orders_with_destination['order_date']
    print(f"Date Range:{dates.iloc[0]} to {dates.iloc[-1]}")
   
    date_range = {
        1: [dt(2016,10,1), dt(2017,2,28)], # 5 months
        2: [dt(2017,3,1), dt(2017,4,30)], # 2 months
        3: [dt(2017,5,1), dt(2017,6,30)], # 2 months
        4: [dt(2017,7,1), dt(2017,8,31)], # 2 months
        5: [dt(2017,9,1), dt(2017,10,31)], # 2 months
        6: [dt(2017,11,1), dt(2017,12,31)], # 2 months
        7: [dt(2018,1,1), dt(2018,2,28)], # 2 months
        8: [dt(2018,3,1), dt(2018,4,30)], # 2 months
        9: [dt(2018,5,1), dt(2018,6,30)], # 2 months
        10: [dt(2018,7,1), dt(2018,8,31)] # 2 months
    }
    NUM_DAYS = len(date_range)

    orders_daywise = date_split(orders_with_destination)
    print("Avg. num_orders:", sum([l.shape[0] for l in orders_daywise])//len(orders_daywise))


    # Orders Data
    for day in range(NUM_DAYS):
        # Real Orders Data
        orders_with_destination = orders_daywise[day]
        orders = pd.merge(orders_with_destination, locs_df, left_on='cust_loc', right_on='loc')
        orders = orders.rename(columns={'node':'cust_node'})
        NUM_ORDERS = orders.shape[0]

        final_orders = pd.merge(orders, rests, on='rest_node')
        # final_orders['deliver_dist'] = final_orders.apply(lambda row:haversine_distance(row['cust_lat'],row['cust_lng'],row['rest_lat'],row['rest_lng']),axis=1)
        final_orders = final_orders[['order_id','customer_id','order_date','cust_lat','cust_lng','cust_loc','cust_node','rest_node','prep_time']]
        final_orders['deliver_dist'] = final_orders.apply(lambda row:apsp[row['cust_node']][row['rest_node']]['dist'], axis=1)
        final_orders['deliver_time'] = final_orders.apply(lambda row:apsp[row['cust_node']][row['rest_node']]['time'], axis=1)
        # packaging time is at most 2 minutes and delivery time is atmost 10 minutes
        final_orders['deliver_dist'] = final_orders['deliver_dist'].apply(lambda x:int(x))
        final_orders['deliver_time'] = final_orders['deliver_time'].apply(lambda x:int(x))

        # choose a placed_ts for each order such that it is placed somewhere between timestamp 0 and NUM_TIMESTAMPS-delivery_time
        final_orders['tot_time'] = final_orders['prep_time'] + final_orders['deliver_time'] 
        final_orders['placed_ts'] = final_orders['tot_time'].apply(lambda x:random.choice(range(NUM_TIMESTAMPS-x))) # unlike other datasets, here we don't ensure that the placed_ts are distinct for each order, which is fine
        final_orders['prep_ts'] = final_orders['placed_ts'] + final_orders['prep_time']
        final_orders['deliver_ts'] = final_orders['prep_ts'] + final_orders['deliver_time']
        final_orders = final_orders.sort_values('order_date').reset_index()

        # breakpoint()
        
        print(final_orders)

        rel_cols = ['order_id','cust_node','rest_node','placed_ts','prep_time','prep_ts','deliver_time','deliver_ts','deliver_dist']
        final_orders = final_orders[rel_cols]

        # SAVING ORDERS.CSV #
        orders_path = os.path.join(data_path, f"Q/orders/_orders_{day+1}.csv")
        final_orders.to_csv(orders_path, index=False)
        print(f"Orders data for day {day+1} stored at {orders_path}")
        
        
        