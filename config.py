class Config(object):
    data_dir = './data/'
    save_dir = './saves/'
    save_name = 'best_model'
    rom_dir = './ROM/'
    actor_checkpoint = './saves/actor/actor_model.ckpt'
    learning_rate = 0.001
    keep_prob = 0.8
    decay_steps = 100000
    anneal_factor = 0.96
    OUTPUT_SIZE = 5
    '''
    Screenshot Dimensions
    '''
    src_w = 640
    src_h = 480
    src_d = 3
    '''
    Sample Image Sizes
    '''
    img_w = 200
    img_h = 66
    img_d = 3
    num_frames = 4
    inp_shape = (None, img_h, img_w, img_d)
    save_freq = 1000
    sum_dir = './summaries/'
    is_training = True
    epochs = 100
    batch_size = 50
    resume_training = False
    initial_epsilon = 1.0
    epsilon_decay = 0.975
    final_epsilon = 0.05
    replay_memory = 500000
    start_memory_sample = 50000
    max_episodes = 500000
    val_split = 0.1

