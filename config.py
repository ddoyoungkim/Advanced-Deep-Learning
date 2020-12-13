class Config():
    def __init__(
        self,
        vocab_size,
        CUDA = 1,
        name = "traj_rnnbased_mapemb_nei_div3",
        
        
        k_near_vocabs = 20,
        n_negatives = 100,
        temp = 100,
        hidden_size=100,
        spatial_input_size=2,
        num_hidden_layers=2,
        num_attention_heads=5,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        activation = 'relu',
        perm_class_num = 2,
        attention_probs_dropout_prob=0.1,
        max_edge_position_embeddings=110,
        max_traj_position_embeddings=120,
        pad_token_id=1,
        del_tasks="dest_mask_perm_aug",
        
        loss_dest_weight=0.01,
        loss_mask_weight=1,
        loss_perm_weight=10,
        
        batch_size = 12000,
        n_trains = 1133657, 
        processors_trains = 36,
        n_vals = 284997,
        processors_vals = 9,
        train_limits = 100,
        val_limits = 1000,
        
        save_epoch = 1,
        
        resume = True,
        path_state = "traj_rnnbased_mapemb_nei_div3_num_hid_layer_2_e1.pt",
        s_epoch = 1,
        **kwargs):
        self.CUDA = CUDA
        
        self.vocab_size = vocab_size
        self.name=name
        self.k_near_vocabs = k_near_vocabs
        self.n_negatives = n_negatives
        self.temp = temp
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.spatial_input_size = spatial_input_size

        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        
        self.activation = activation
        self.perm_class_num = perm_class_num
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_edge_position_embeddings = max_edge_position_embeddings
        self.max_traj_position_embeddings = max_traj_position_embeddings
        self.pad_token_id = pad_token_id
        self.del_tasks=del_tasks
        
        self.loss_dest_weight = loss_dest_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_perm_weight = loss_perm_weight
        
        self.batch_size = batch_size
        self.n_trains = n_trains
        self.n_vals = n_vals
        self.processors_trains = processors_trains
        self.processors_vals = processors_vals
        
        self.save_epoch = save_epoch
        self.train_limits = train_limits
        self.val_limits = val_limits
        
        self.resume=resume
        self.path_state=path_state
        self.s_epoch=s_epoch
        
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count