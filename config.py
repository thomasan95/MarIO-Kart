class Config(object):
    data_dir = './data/'
    save_dir = './saves/'
    save_name = 'best_model'
    learning_rate = 0.001
    keep_prob = 0.8
    batch_size = 32
    decay_steps = 100000
    anneal_factor = 0.96
    OUTPUT_SIZE = 5
    img_w = 200
    img_h = 66
    img_d = 3
    save_freq = 1000
    sum_dir = './summaries/'
    is_training = True
    