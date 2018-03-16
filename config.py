class Config(object):
    """
    Configuration file for specifying parameters for the executable files
    """
    data_dir = './data/'
    save_dir = './saves/'
    save_name_supervised = 'supervised/best_model_supervised'
    save_name_reinforcement = 'reinforcement/best_model_reinforcement'
    first_reinforcement = True
    rom_dir = './ROM/'
    pickle_dir = './pickles/'
    checkpoint = './saves/actor/actor_model.ckpt'
    learning_rate = 0.001
    keep_prob = 0.9
    decay_steps = 100000
    anneal_factor = 0.95
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
    r_inp_shape = (None, img_h, img_w, img_d*num_frames)
    save_freq = 1000
    sum_dir = './summaries/'
    is_training = True
    epochs = 200
    batch_size = 50
    resume_training = True
    if first_reinforcement:
        initial_epsilon = 0.9
    else:
        initial_epsilon = 0.5
    epsilon_decay = 0.975
    final_epsilon = 0.05
    replay_memory = 50000
    start_memory_sample = 50000
    max_episodes = 500000
    val_split = 0.1
    shuffle = True
