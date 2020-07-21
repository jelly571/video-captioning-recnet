import os
import time


class FeatureConfig:
    models = [ "MSVD_InceptionV4" ]
    size = 0
    for model in models:
        if 'InceptionV4' in model:
            size += 1536
        else:
            raise NotImplementedError("Unknown model: {}".format(model))


class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 }
    embedding_size = 468


class MSVDLoaderConfig:
    train_caption_fpath = "data/MSVD/metadata/train.csv"
    val_caption_fpath = "data/MSVD/metadata/val.csv"
    test_caption_fpath = "data/MSVD/metadata/test.csv"
    min_count = 1
    max_caption_len = 30

    phase_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    frame_max_len = 300 // 5
    frame_sample_len = 28

    num_workers = 4


class MSRVTTLoaderConfig:
    train_caption_fpath = "data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "data/MSR-VTT/metadata/test.json"
    min_count = 1
    max_caption_len = 30

    phase_video_feat_fpath_tpl = "data/{}/features/{}_{}.hdf5"
    frame_sampling_method = 'uniform'; assert frame_sampling_method in [ 'uniform', 'random' ]
    frame_max_len = 300 // 5
    frame_sample_len = 28

    num_workers = 4


class DecoderConfig:
    rnn_type = 'LSTM'; assert rnn_type in [ 'LSTM', 'GRU' ]
    rnn_num_layers = 1
    rnn_num_directions = 1; assert rnn_num_directions in [ 1, 2 ]
    rnn_hidden_size = 512
    rnn_attn_size = 256
    rnn_dropout = 0.5
    rnn_teacher_forcing_ratio = 1.0


class GlobalReconstructorConfig:
    type = 'global'
    rnn_type = 'LSTM'
    rnn_num_layers = 1
    rnn_num_directions = 1; assert rnn_num_directions in [ 1, 2 ]
    rnn_hidden_size = FeatureConfig.size
    rnn_dropout = 0.5


class LocalReconstructorConfig:
    type = 'local'
    rnn_type = 'LSTM'
    rnn_num_layers = 1
    rnn_num_directions = 1; assert rnn_num_directions in [ 1, 2 ]
    rnn_hidden_size = FeatureConfig.size
    rnn_attn_size = 256
    rnn_dropout = 0.5


class TrainConfig:
    corpus = 'MSVD'; assert corpus in [ 'MSVD', 'MSR-VTT' ]
    reconstructor_type = 'global'; assert reconstructor_type in [ 'global', 'local' ]

    feat = FeatureConfig
    vocab = VocabConfig
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig,
    }[corpus]
    decoder = DecoderConfig
    reconstructor = {
        'global': GlobalReconstructorConfig,
        'local': LocalReconstructorConfig,
    }[reconstructor_type]


    """ Optimization """
    epochs = {
        'MSVD': 30,
        'MSR-VTT': 30,
    }[corpus]
    batch_size = 200
    shuffle = True
    optimizer = "AMSGrad"
    gradient_clip = 5.0 # None if not used
    lr = {
        'MSVD': 2e-5,
        'MSR-VTT': 4e-5,
    }[corpus]
    lr_decay_start_from = 0
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    weight_decay = 1e-5
    recon_lambda = {
        'global': 0.2,
        'local': 0.1,
    }[reconstructor.type]
    reg_lambda = 0.

    """ Pretrained Model """
    pretrained_decoder_fpath = "checkpoints/RecNet | MSVD | FEAT MSVD_InceptionV4 mcl-30 | EMB 468 | DEC uni-LSTM-l1-h512 at-256 | OPTIM AMSGrad lr-5e-05-dc-20-0.5-5-wd-1e-05 reg-0.0 rec-0.0 | 200720-08:24:54/best.ckpt"
    pretrained_reconstructor_fpath = None

    """ Evaluate """
    metrics = [ 'Bleu_4', 'CIDEr', 'ROUGE_L' ]

    """ ID """
    exp_id = "RecNet-{}".format(reconstructor.type)
    feat_id = "FEAT {} mcl-{}".format('+'.join(feat.models), loader.max_caption_len)
    embedding_id = "EMB {}".format(vocab.embedding_size)
    decoder_id = "DEC {}-{}-l{}-h{} at-{}".format(
        ["uni", "bi"][decoder.rnn_num_directions-1], decoder.rnn_type,
        decoder.rnn_num_layers, decoder.rnn_hidden_size, decoder.rnn_attn_size)
    reconstructor_id = "REC {}-{}-l{}-h{}".format(
        ["uni", "bi"][reconstructor.rnn_num_directions-1], reconstructor.rnn_type, reconstructor.rnn_num_layers,
        reconstructor.rnn_hidden_size)
    if reconstructor.type == 'local':
        reconstructor_id += " at-{}".format(reconstructor.rnn_attn_size)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{} reg-{} rec-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay, reg_lambda, recon_lambda)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = " | ".join([ exp_id, corpus, feat_id, embedding_id, decoder_id, reconstructor_id, optimizer_id, timestamp ])

    """ Log """
    log_dpath = "logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/decoder/cross_entropy"
    tx_train_reconstruction_loss = "loss/train/reconstructor"
    tx_train_entropy_loss = "loss/train/decoder/entropy"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/decoder/cross_entropy"
    tx_val_reconstruction_loss = "loss/val/reconstructor"
    tx_val_entropy_loss = "loss/val/decoder/entropy"
    tx_lr = "params/lr"


