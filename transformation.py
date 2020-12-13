#%%
import random
import numpy as np
import torch
from torch_geometric.utils import subgraph
from collections import defaultdict
import pickle
import traceback

# REF: traj 정보는 Data.edge_attr에 먼저 positional encoding 해줌.
# %%
class Normal(object):
    def __call__(self, data):
        
        data, _ = _subgraph(data)

        return data
    
    
#%%
class Reversed(object):
    """
    On the probability of p, reverse the trajectory
        Overwrite the trajectory edge.
        (1) reverse the trajectory order in edge_attribute 
            by calculating max_length + 1 - original number in edge_attribute
        (2) switch the order of edge
            (start_node, end_node) -> (end_node, start_node)
        (3) delete duplicated edges 
    """
    def __init__(self, p = 0.5):
        self.p = p
        
    def __call__(self, data):
        # data.clone()
        data1, _ = _subgraph(data.clone())
        data2, order2index2 = _subgraph(data.clone())
        # Anchor
        data1.y = torch.tensor(-1, dtype=torch.long).unsqueeze(-1)
        # Positive sample
        data2.y = torch.tensor(0, dtype=torch.long).unsqueeze(-1)
        # Negative sample
        data3 = data2.clone()
        data3.y = torch.tensor(0, dtype=torch.long).unsqueeze(-1)
        if random.random() > self.p :
            data3.y = torch.tensor(1, dtype=torch.long).unsqueeze(-1)
            edge_attribute = data3.edge_attribute.clone()
            edge_index = data3.edge_index.clone()
            tm_index = data3.tm_index.clone()
            length = len(order2index2)

            ## (1) reverse the trajectory order in edge_attribute and tm_index
            edge_attribute = torch.flip(edge_attribute,[-1])
            tm_index = torch.cat((torch.flip(tm_index[:len(data3.traj_vocabs)], dims=(0,)),\
                                     tm_index[len(data3.traj_vocabs):]))

            ## (2) switch the order of edge
            edge_index[:, edge_attribute] = \
            torch.stack([edge_index[1, edge_attribute],\
                edge_index[0, edge_attribute]], dim = 0 )

            ## (3) delete the duplicated edge_index
            edge_index_unique, inv, dup_cnt = torch.unique(edge_index, dim = 1, return_inverse=True, return_counts=True)

            # traj2attr = dict(zip(list(range(1,len(edge_attribute)+1)), edge_attribute.tolist()))
            # old2new = dict(zip(edge_attribute.tolist(), inv[edge_attribute].tolist()))

            ## (4) find the duplicated edge_index and reverse it again (keep the number of edge_index)
            duplicated_edge_index = edge_index[:,(dup_cnt-1).nonzero().squeeze(1)]
            duplicated_edge_index = duplicated_edge_index[[1,0],:]

            ## (5) concat edge_index_unique and duplicated_edge_index
            edge_index = torch.cat((edge_index_unique, duplicated_edge_index), dim = 1)

            ## (6) update the edge_index, edge_attribute, edge_attribute_len
            data3.edge_index = edge_index # torch.tensor(list(traj_edgedict.keys()),dtype=torch.long).transpose(0,1)           
            data3.edge_attribute = inv[edge_attribute].to(torch.long) # torch.tensor(list(traj_edgedict.values()),dtype=torch.long)
            data3.edge_attribute_len = torch.tensor(len(data3.edge_attribute), dtype=torch.long).unsqueeze(-1)
            data3.tm_index = tm_index

            

        return data1, data2, data3
        
#%%
class Permuted(object):
    """
    Divide one trajectory into two trajectories.
    On the probability of p, reverse one of two trajectories.
        Overwrite the trajectory edge.
        (1) reverse the trajectory order in edge_attribute 
            by calculating max_length + 1 - original number in edge_attribute
        (2) switch the order of edge
            (start_node, end_node) -> (end_node, start_node)
        (3) delete duplicated edges 
    """
    def __init__(self, p1 = 0.5):
        '''
        p1 : permuted or not
        '''
        self.p1 = p1

    def __call__(self, data):
        # data.clone()
        data1, _ = _subgraph(data.clone())
        data2, order2index2 = _subgraph(data.clone())
        if random.random() > self.p1 :
            data1.y = torch.tensor(0, dtype=torch.long).unsqueeze(-1)
            data2.y = torch.tensor(0, dtype=torch.long).unsqueeze(-1)
            return data1, data2
        else :
            edge_attribute = data2.edge_attribute.clone()
            edge_index = data2.edge_index.clone()
            length = len(order2index2)
            ## (1) choose two traj numbers
            no = random.randint(1,length-1) # edge_attribute start at 0
            nos = list(range(no,random.randint(-1,no-2),-1)) # set the length of the permuted traj
            no_edge1, no_edge2 = edge_index[:,edge_attribute[nos[0]]].clone()
            # print(no, nos)
            # print(edge_attribute)
            ## (2) switch the order of nodes
            ## [no2_edge1, no2_edge2=no1_edge1], [no2_edge=no1_edge1, no1_edge2]
            ## [no2_edge1, no1_edge2] [ no1_edge2, no2_edge2=no1_edge1]
            edge_index[:,edge_attribute[nos[:-1]]] = \
                torch.stack((edge_index[1,edge_attribute[nos[:-1]]],\
                    edge_index[0,edge_attribute[nos[:-1]]]), dim = 0 )
            edge_index[1,edge_attribute[nos[-1]]] = no_edge2
            edge_attribute[nos[:-1]] = torch.flip(edge_attribute[nos[:-1]],[-1])
            # print(edge_attribute)
            ## (3) delete duplicated edges 
            edge_attribute_long = torch.zeros(edge_index.size(1), dtype=torch.long)
            edge_attribute_long[edge_attribute] = torch.arange(1,length+1)
            edge_attribute_long = edge_attribute_long.unsqueeze(1)
            key_value = torch.cat([edge_index,edge_attribute_long.view(1,-1)],dim=0)
            key_value = key_value[:,key_value[2,:].sort()[1]]
            traj_edgedict = dict(zip(list(map(tuple,key_value[[0,1],:].transpose(1,0).tolist())),key_value[-1,:].tolist()))
            traj_edgedict = dict(sorted(traj_edgedict.items()))

            data2.edge_index = torch.tensor(list(traj_edgedict.keys()),dtype=torch.long).transpose(0,1)           
            
            edge_attribute = torch.tensor(list(traj_edgedict.values()),dtype=torch.long)                
            traj_orderdict = dict(zip(list(edge_attribute[edge_attribute.nonzero()]), list(edge_attribute.nonzero())))
            traj_orderdict = dict(sorted(traj_orderdict.items()))
            
            data2.edge_attribute = torch.tensor(list(traj_orderdict.values()),dtype=torch.long)
            data2.edge_attribute_len = torch.tensor(len(data2.edge_attribute), dtype=torch.long).unsqueeze(-1)

            data1.y = torch.tensor(1,dtype=torch.long ).unsqueeze(-1)
            data2.y = torch.tensor(1,dtype=torch.long ).unsqueeze(-1)
        # edge_attribute = edge_attribute[edge_attribute.nonzero()[:,0]].view(-1)
        return data1, data2

    # def __repr__(self):
    #     return "permuted"

#%%
class Masked(object):
    """
    Corrupt one random node.
    (1) find random trajectory number(N) in edge_attribute
    (2) pivot node = start node
    (3) Replace the pivot node with masked token in edge_index
    
    Update 12/4 : delete the connected nodes to the node will be masked.

    """
    def __init__(self, p = 0.2):
        """
        masking = 20 %
        """
        self.p = p

    def __call__(self, data):
        assert len(data.x) > 0
        data, order2index = _subgraph(data.clone())

        edge_attribute = data.edge_attribute.clone()
        edge_index = data.edge_index.clone()#[:,:1560]
        length = len(edge_attribute)
        # print("origin edge_attribute\n", edge_attribute)
        # print("origin edge_index\n", edge_index[:,edge_attribute])
        ## (1) find random trajectory number(N) in edge_attribute
        # get the random trajectory numbers
        nos = torch.randint(low = 1, high = length-1, size=(round(length * self.p),), dtype=torch.long)
        ## (2) Update data.y and update the masked node to "3" in data.x and
        j = edge_index[0,edge_attribute[nos]] # index_of_edge_index -> edge_index's j:node_index
        ### if actual vocab_index is 0, then remove it
#         for i in range(len(j)) :
#             if data.x[j[i]] == torch.tensor(0, dtype = torch.long):
#                  j.remove(j[i])
        j = j[(data.x[j] != 0).squeeze()]
        if j.nelement() == 0:
            return None
        
        data.y = data.x[j].unique(sorted=True).squeeze().detach().clone() # actual vocab_index 
#         data.y_len = torch.tensor(len(data.y), dtype=torch.long).unsqueeze(-1)
        # if data.y == torch.tensor(0, dtype = torch.long):
        #     return None
        #data.y = torch.tensor(j.detach().clone(), dtype=torch.long) # j=y: node_index in data.x
        data.x[j] = torch.tensor(3, dtype = torch.long)
        ## (3) delete the edges in edge_index that are connected to the masked node. 
        ## find the incomming/outgoing nodes to the masked node
        ### keep the original index in new tensor 
        edge_index_index = torch.arange(1, edge_index.size(1)+1, dtype=torch.long)
        # print("edge_indx_index1\n",edge_index_index)
        ### find the index of incoming and outgoing edge index using new tensor
        #### find the index of incoming and outgoing edge index
        # print("j\n",j)
        incoming_nodes_index = (edge_index[1,:][...,None] == j).any(-1).nonzero().squeeze()
        # print("incoming_nodes_index\n", incoming_nodes_index)
        # print(edge_index[:,incoming_nodes_index])
        outgoing_nodes_index = (edge_index[0,:][...,None] == j).any(-1).nonzero().squeeze()
        # print("ougoing_nodes_index\n", outgoing_nodes_index)
        # print(edge_index[:,outgoing_nodes_index])
        #### update the index tensor with 0 if edges are incoming/outgoing edge to/from masked node
        # print(incoming_nodes_index)
        # print(edge_index_index.size())
        # print(edge_index_index)
        edge_index_index[incoming_nodes_index] = 0
        edge_index_index[outgoing_nodes_index] = 0
        # print("edge_index_index2\n", edge_index_index)
        edge_index_index[edge_attribute] = edge_attribute + 1 # edge_attribute starts at 0, but edge_index_index starts at 1
        # print("edge_index_index3\n", edge_index_index)
        #### remove the incoming/outgoing edge to/from masked node
        edge_index = edge_index[:,edge_index_index.nonzero().squeeze()]
        # print("edge_index_index4\n", edge_index_index)
        #### Update edge_attribute using edge_index_index
        # edge_index_index = edge_index_index[edge_index_index.nonzero().squeeze()] - 1 # edge_attribute starts at 0, but edge_index_index started at 1
        # edge_attribute = torch.tensor([(edge_index_index==old).nonzero().squeeze().item() for old in edge_attribute], dtype=torch.long)
        edge_index_index[edge_index_index.nonzero().squeeze()] = torch.arange(0,edge_index.size(1),dtype=torch.long) # edge_attribute starts at 0, but edge_index_index started at 1
        edge_attribute = edge_index_index[edge_attribute]
        # print("edge_attribute2\n",edge_attribute)
        # print("edge_index2\n",edge_index[:,edge_attribute])
        # import sys;sys.exit()
        data.edge_index = edge_index
        data.edge_attribute = edge_attribute
        # import sys; sys.exit()
        return data
#%%
# target node의 인접 노드 중에서 랜덤으로 하나 골라서 옮기기
class Augmented(object):
    """
    Replace one random node to the neighbor node
    (1) find random trajectory number(N) in edge_attribute
    (2) pivot node = start node
    (3) get the previous and next nodes of pivot node
    (4) get the neighbor nodes of them respectively
    (5) get the common nodes of them
    (6) if len(common_nodes) > 1 : choose the node randomly; break while
    (7) change the trajectory numbers of [the previous node, new node] and [new node, the next node] in edge_attribute into N and N-1
    TODO: Whose neighbor nodes?
    """
    def __call__(self, data):
        assert len(data.x) > 0
        data, order2index = _subgraph(data)
        data1 = data.clone() # original
        data2 = data.clone() # augmented

        edge_attribute = data2.edge_attribute.clone()
        edge_index = data2.edge_index.clone()

        length = len(order2index) # get the length of trajectory(duplicated removed)
        
        i_firstwhile = 0
        while True:
            i_firstwhile += 1
            if i_firstwhile > length*5:
                #print("Trans firstwhile", i)
                return None
            ## (1) find random trajectory number(N) in edge_attribute
            ## (2) pivot node = start node
            no = random.randint(1,length-2) # second node ~ third node from behind #TODO : third? or second?
            # Get the neighbor nodes 
            ## (3) get the previous and next node to get the node that will be augmented instead
            prev_node = edge_index[0,edge_attribute[no-1]].clone().item()
            pivot_node = edge_index[0,edge_attribute[no]].clone().item()
            next_node = edge_index[1,edge_attribute[no]].clone().item()

            #  (4) get the neighbor nodes of them respectively
            ## (4-1) neighbors of the previous node
            ## (4-2) neighbors of the next node
            try :
                neighbor_nodes1 = edge_index[1,(edge_index[0]==prev_node).nonzero().squeeze()]
                neighbor_nodes2 = edge_index[0,(edge_index[1]==next_node).nonzero().squeeze()]
            
                ## (5) get the common nodes of them
                indices = torch.zeros_like(neighbor_nodes1, dtype = torch.bool)
                for elm in neighbor_nodes2 :
                    indices = indices | (neighbor_nodes1 == elm)
                common_nodes = neighbor_nodes1[indices]
            except TypeError:
                continue
            ## (6) if len(common_nodes) > 1 : choose the node randomly; break while
            if len(common_nodes) > 2:
                # Choose one node to replace the original node
                i_secondwhile = 0
                while True:
                    i_secondwhile += 1
                    if i_secondwhile > len(common_nodes)*5:
                        #print("Trans secondwhile", i)
                        return None
                    augment_node = common_nodes[random.randint(0,len(common_nodes)-1)].item()
                    if augment_node != pivot_node:
                        break
                break
        
        # print(prev_node, pivot_node, next_node)
        # print(edge_index[:,edge_attribute[no-1]],edge_index[:,edge_attribute[no]])
        # print(augment_node)

        ## (7) change the trajectory numbers of [the previous node, new node] and [new node, the next node] in edge_attribute into N and N-1
        # Modify two trajectories' edges
        # modify edge_attribute of [prev_node, augment_node]:no-1 [augment_node,next_node]:no [prev_node,original_node]: 0
        ## Get the index of the [prev_node, augment_node] in edge_index
        idx1 = (((edge_index.transpose(1,0) == torch.tensor([prev_node, augment_node])).to(torch.long).prod(dim=1))).nonzero().squeeze().item()
        ## Get the index of the [augment_node,next_node] in edge_index
        idx2 = (((edge_index.transpose(1,0) == torch.tensor([augment_node,next_node])).to(torch.long).prod(dim=1))).nonzero().squeeze().item()
        # idx = (edge_index.transpose(1,0) == torch.Tensor([augment_node,next_node],dtype=torch.long)).nonzero()[0][0]
        
        # # Check the result
        # print("idx1",edge_index[:,idx1])
        # print("idx2",edge_index[:,idx2])
        
        edge_attribute[no - 1] = idx1 # edge_attribute's index starts at 0.
        edge_attribute[no] = idx2 # edge_attribute's index starts at 0.
       
        # print(edge_index[:,edge_attribute[no-1]],edge_index[:,edge_attribute[no]])
        data2.edge_attribute = edge_attribute
        data2.edge_attribute_len = torch.tensor(len(data2.edge_attribute), dtype=torch.long).unsqueeze(-1)
        data2.x[pivot_node] = torch.tensor(augment_node, dtype=torch.long)

        data2.y = torch.tensor(no, dtype=torch.long).unsqueeze(-1) # index of the edge_atrr, starting from 0.
        data1.y = torch.tensor(no, dtype=torch.long).unsqueeze(-1) # index of the edge_atrr, starting from 0.
        return data1, data2

#%%
class Destination(object):
    """
    return the last node of the trjaectory
    last node : second node of the last edge attribute
    """
    def __init__(self, p =0.9):
        self.p = p
    def __call__(self, data):
        assert len(data.x) > 0
        data, order2index = _subgraph(data)

        edge_attribute = data.edge_attribute.clone()
        edge_index = data.edge_index.clone()
        length = len(order2index)
        
        # order2node = dict(zip(list(range(1,length+2)),edge_index[0,edge_attribute].tolist() + [edge_index[1,edge_attribute[-1]].item()]))
        last_edge_attribute = edge_attribute[-1]
        edge_attribute = edge_attribute[:int(length * self.p)]
        
        data.edge_attribute = edge_attribute
        data.edge_attribute_len = torch.tensor(len(data.edge_attribute), dtype=torch.long).unsqueeze(-1)
        data.y = torch.tensor(edge_index[1,last_edge_attribute].item(), dtype=torch.long).unsqueeze(-1)
        # data.y = torch.tensor(order2node[length+1], dtype=torch.long).unsqueeze(-1)
        if data.x[data.y] == torch.tensor(0, dtype=torch.long) :
            return None
        ############post processing########
        edge_attr = torch.zeros(data.edge_index.size(1), dtype=torch.long)
        edge_attr[data.edge_attribute] = torch.arange(1,data.edge_attribute.size(0)+1)
        # traj_nodes_idx
        remain_nodes_indice = torch.cat((data.edge_index[:,data.edge_attribute][0],
                                         data.edge_index[:,data.edge_attribute][[1],-1]), dim=0)
        
        # (node_index that are connected to traj_nodes: <= E,1)
        active_edge_indice = (data.edge_index.unsqueeze(-1) == remain_nodes_indice).any(-1).any(0).nonzero()
        # (2, #remain_node_indice) : edge index of (traj nodes + conn nodes)
        remain_edge_indice = data.edge_index[:,active_edge_indice.squeeze()]
        # get (traj nodes + conn nodes)' edges
        _remain_edge_indice = (data.edge_index.unsqueeze(-1) == torch.unique(remain_edge_indice.view(-1))).any(-1).any(0).nonzero()
        # get new edge_index & edge_attribute
        new_edge_index = data.edge_index[:,_remain_edge_indice.squeeze()]
        new_edge_attr = edge_attr[_remain_edge_indice]
        data.edge_index = new_edge_index
        data.edge_attribute = torch.argsort(new_edge_attr.squeeze(), descending=False)[-new_edge_attr.nonzero().size(0):]
        data.edge_attribute_len = torch.tensor(len(data.edge_attribute), dtype=torch.long).unsqueeze(-1)
        
        unique_out, inv_indices = torch.unique_consecutive(data.tm_index[:data.traj_len],
                                                           return_inverse=True)
        # new traj part for tm_index
        tm_traj = unique_out[:remain_nodes_indice.size(0)][inv_indices[inv_indices<remain_nodes_indice.size(0)]]
        # only conn nodes 
        tm_conn_nodes = torch.tensor(list(set(torch.unique(new_edge_index.view(-1)).numpy()) -\
                                       set(remain_nodes_indice.numpy())))
        if tm_conn_nodes.nelement() == 0:
            return None
        data.tm_index = torch.cat((tm_traj, tm_conn_nodes[torch.randperm(tm_conn_nodes.size(0))]), dim=0)
        data.tm_len = torch.tensor(len(data.tm_index), dtype=torch.long).unsqueeze(-1)
        data.traj_len = torch.tensor(len(tm_traj), dtype=torch.long).unsqueeze(-1)
        return data

def _subgraph(data):

    x = data.x.clone()
    edge_index = data.edge_index.clone()
    __edge_attr = data.edge_attribute.clone()
    traj_vocabs = data.traj_vocabs.clone()
    traj_index = torch.tensor([(x==n).nonzero().squeeze()[0].item() for n in traj_vocabs])
    
    order2index=defaultdict(list)
    for i, idx  in enumerate(__edge_attr,1):
        order2index[i] = int(idx)

    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
    edge_attr[__edge_attr] = torch.arange(1, __edge_attr.size(0)+1)

    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    inds = torch.unique(traj_index)
    mask[inds] = True
    perm = torch.randperm(torch.sum(~mask.squeeze()))
    conn = torch.arange(mask.size(0))[~mask.squeeze()][perm[:max(3, len(inds)//3)]]
#     conn = torch.tensor([], dtype=torch.long)

    nodes = torch.cat((inds, conn), dim=0)
    edge_ind, edge_att = subgraph(nodes, edge_index, edge_attr, num_nodes=len(x))

    # edge_attr matching between origin and subgraph
    edge_attr = torch.argsort(edge_attr,descending=False)[-edge_attr.nonzero().size(0):]
    edge_att = torch.argsort(edge_att,descending=False)[-edge_att.nonzero().size(0):]
    origin_sub = {int(p):int(c) for p,c in zip(edge_attr, edge_att)}

    edge_att = torch.tensor([origin_sub[index] for order, index in order2index.items()],
                            dtype=torch.long)

    tm_index = torch.cat((torch.tensor(traj_index, dtype=torch.long),
                                        conn.to(torch.long)), dim=0)

    data.edge_index = edge_ind.to(torch.long)
    data.edge_attribute = edge_att
    data.edge_attribute_len = torch.tensor(len(edge_att), dtype=torch.long).unsqueeze(-1)
    data.tm_index = tm_index
    data.tm_len = torch.tensor(len(tm_index), dtype=torch.long).unsqueeze(-1)

    return data, order2index
#%%
# data = TrajData(x=torch.tensor(all_nodes, dtype=torch.long).unsqueeze(1), 
#         edge_index=edge_ind.to(torch.long),
#         edge_attribute=edge_att,
#         edge_attribute_len=torch.tensor(len(edge_att), dtype=torch.long).unsqueeze(-1),
#         tm_index = tm_index,
#         traj_vocabs = torch.tensor(traj_nodes,dtype=torch.long), 
#         traj_len= torch.tensor(len(traj_index), dtype=torch.long).unsqueeze(-1),
#         tm_len= torch.tensor(len(tm_index), dtype=torch.long).unsqueeze(-1)
#         )
