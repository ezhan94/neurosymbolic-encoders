import argparse
import json
import os
import random
from comet_ml import Experiment

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import time
from time import gmtime, strftime

from lib.models import get_model_class
from datasets import load_dataset


def main(args, config_files, save_dir, logger):
    # Online logging only works for a single config file right now.
    for config_file in config_files:

        # Load JSON config file
        with open(os.path.join(config_dir, config_file), 'r') as f:
            config = json.load(f)

        trial_id = config_file[:-5] # remove .json at the end
        print('########## Trial {}:{} ##########'.format(exp_name, trial_id))

        # Create save folder
        save_path = os.path.join(save_dir, trial_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(os.path.join(save_path, 'checkpoints')) # for model checkpoints
            os.makedirs(os.path.join(save_path, 'results')) # for saving various results afterwards (e.g. plots, samples, etc.)

        logger.experiment.log_parameters(config)

        data_config, model_config, train_config, summary = start_training(
            save_path=save_path,
            data_config=config['data_config'],
            model_config=config['model_config'],
            train_config=config['train_config'],
            gpus=args.gpus,
            logger = logger
        )

        # Save config file (for reproducability)
        config['data_config'] = data_config
        config['model_config'] = model_config
        config['train_config'] = train_config
        print(config)
        os.makedirs(os.path.join(save_dir, 'configs'), exist_ok=True)
        with open(os.path.join(save_dir, 'configs', config_file), 'w+') as f:
            json.dump(config, f, indent=4)

        # Save entry in master file
        summary['log_path'] = os.path.join(args.save_dir, exp_name, trial_id)
        master['summaries'][trial_id] = summary

        print(master)

        # Save master file
        with open(os.path.join(save_dir, 'master.json'), 'w') as f:
            json.dump(master, f, indent=4)


def start_training(save_path, data_config, model_config, train_config, gpus, logger):
    summary = { 'training' : [] }

    # Sample and fix a random seed if not set in train_config
    if 'seed' not in train_config:
        train_config['seed'] = random.randint(0, 9999)
    seed = train_config['seed']
    pl.seed_everything(seed = seed)
    torch.backends.cudnn.deterministic = True

    # Initialize dataset
    datamodule = load_dataset(data_config)
    datamodule.setup()
    
    summary['dataset'] = datamodule.summary

    # Add state and action dims to model config
    model_config['state_dim'] = datamodule.state_dim
    model_config['action_dim'] = datamodule.action_dim

    # Get model class
    model_class = get_model_class(model_config['name'].lower())

    # Initialize model
    model = model_class(model_config)
    summary['model'] = model_config

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_path, 'checkpoints'),
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    start_time = time.time()

    if 'max_progs' not in train_config or train_config['max_progs'] == 0:
        iterate_counter = 0
        logger_metadata = {'iterate_counter': iterate_counter}

        model = train_model(model, gpus, train_config, logger, checkpoint_callback, datamodule, model_config, logger_metadata)

    elif 'max_progs' in train_config and train_config['max_progs'] > 0:
        assert 'near_config' in train_config
        assert model.has_programmatic_encoder
        assert hasattr(model, 'num_progs')

        iterate_counter = -1

        while model.num_progs_fixed < train_config['max_progs']:
            program_num = model.num_progs_fixed
            iterate = True

            iterate_counter += 1
            logger_metadata = {'iterate_counter': iterate_counter, 'program_num': program_num}

            # Train model with new fully neural encoder program (enc_progs_learn)
            model = train_model(model, gpus, train_config, logger, checkpoint_callback, datamodule, model_config, logger_metadata)

            torch.save(model, os.path.join(save_path, 'checkpoints/model_iter_' + str(iterate_counter)))

            while iterate:
                # TODO potentially a bug with pytorch lightning, need to turn requires_grad back on
                # See https://github.com/PyTorchLightning/pytorch-lightning/issues/2540
                for param in model.parameters():
                    param.requires_grad = True

                iterate_counter += 1
                logger_metadata = {'iterate_counter': iterate_counter, 'program_num': program_num}

                is_symbolic = update_neurosymbolic_encoder(model, datamodule, data_config, 
                    train_config['near_config'].copy(), gpus, logger, save_path, logger_metadata)
                iterate = not is_symbolic

                # Train model with new program structure
                model = train_model(model, gpus, train_config, logger, checkpoint_callback, datamodule, model_config, logger_metadata)

                torch.save(model, os.path.join(save_path, 'checkpoints/model_iter_' + str(iterate_counter)))

                if is_symbolic:
                    # Move program from learn to fixed
                    model.enc_progs_fixed.append(model.enc_progs_learn[0])
                    del model.enc_progs_learn[0]
                
            # Log current encoder programs
            if model.z_prog_type == "discrete":
                assert model.num_progs_learn == 0

                program_str = "\n"
                for i in range(model.num_progs):
                    # Compute label distribution
                    labels_list = []
                    for batch in datamodule.train_dataloader():
                        batch_labels, _ = model.get_labels(batch)
                        labels_list.append(batch_labels[i].long())
                    labels = torch.cat(labels_list, axis=0).numpy()

                    label_dist = list(np.bincount(labels.squeeze())/len(labels))
                    while len(label_dist) < int(model.n_clusters/model.num_progs):
                        label_dist.append(0.0)

                    program_str += f"ENCODER PROGRAM {i}: {model.enc_progs_fixed[i].to_str(include_params=True)}\nclass distribution {label_dist}\n"
                
                print(program_str)
                logger.experiment.log_text(program_str, metadata=logger_metadata)

    # Save full pytorch model
    torch.save(model, os.path.join(save_path, 'checkpoints/final_model'))
    logger.experiment.log_model("Prog_Model", os.path.join(save_path, 'checkpoints'), metadata=logger_metadata)        

    summary['total_time'] = round(time.time()-start_time, 3)

    return data_config, model_config, train_config, summary


def train_model(model, gpus, train_config, logger, checkpoint_callback, datamodule, model_config, logger_metadata):
    num_epochs = train_config['num_epochs_init'] if logger_metadata['iterate_counter'] == 0 else train_config['num_epochs_iter']
    
    model.init_model()

    trainer = pl.Trainer(gpus=gpus, max_epochs=num_epochs,
        logger=logger, gradient_clip_val=train_config['clip'], 
        progress_bar_refresh_rate=50, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # Log loss metrics to text.
    loss_log_string = ""
    for metric_key, metric_val in trainer.logged_metrics.items():
        loss_log_string += f"{metric_key}: {metric_val}\n"
    logger.experiment.log_text(loss_log_string, metadata=logger_metadata)

    return model


def update_neurosymbolic_encoder(model, datamodule, data_config, train_config, gpus, 
                            logger, save_path, logger_metadata):
    import torch.nn as nn
    import torch.optim as optim

    from near.algorithms import MC_SAMPLING, IDDFS_NEAR
    from near.eval import test_set_eval
    from near.dsl import import_dsl
    from near.program_graph import ProgramGraph, ProgramNode
    from near.utils.evaluation import label_correctness, mse_eval
    from near.utils.data import prepare_datasets

    train_data, train_labels = extract_labels_from_model(model, datamodule.train_dataloader())
    val_data, val_labels = extract_labels_from_model(model, datamodule.val_dataloader())
    test_data, test_labels = extract_labels_from_model(model, datamodule.test_dataloader())

    batched_trainset, validset, testset = prepare_datasets(
        train_data, val_data, test_data, train_labels, val_labels, test_labels, 
        normalize=train_config['normalize_data'], batch_size=data_config['batch_size'], cast_long=(model.z_prog_type == "discrete"))

    if model.z_prog_type == "discrete":
        lossfxn = nn.CrossEntropyLoss() if model.z_prog_dim != 1 else nn.BCEWithLogitsLoss()
        evalfxn = label_correctness
        num_labels = 2 if model.z_prog_dim == 1 else model.z_prog_dim

        cluster_dist = list(np.bincount(train_labels.squeeze())/len(train_labels))
        while len(cluster_dist) < num_labels:
            cluster_dist.append(0.0)
        print(f"\nTRAIN cluster distribution: {cluster_dist}\n")
        logger.experiment.log_text(f"TRAIN cluster distribution: {cluster_dist}", metadata=logger_metadata)

    elif model.z_prog_type == "continuous":
        lossfxn = nn.MSELoss()
        evalfxn = mse_eval
        num_labels = 1

        print(f"\nTRAIN label (mean, std): ({train_labels.mean():.4f}, {train_labels.std():.4f})\n")
        logger.experiment.log_text(f"TRAIN label (mean, std): ({train_labels.mean():.4f}, {train_labels.std():.4f})", metadata=logger_metadata)

    current_str = f"CURRENT PROGRAM: {model.enc_progs_learn[0].to_str(include_params=True)}\n"
    print(current_str)
    logger.experiment.log_text(current_str, metadata=logger_metadata)

    device = 'cuda:0' if gpus > 0 else 'cpu'
    if device != 'cpu':
        lossfxn = lossfxn.cuda()

    train_config['optimizer'] = optim.Adam
    train_config['lossfxn'] = lossfxn
    train_config['evalfxn'] = evalfxn
    train_config['num_labels'] = num_labels

    DSL_DICT, CUSTOM_EDGE_COSTS = import_dsl(train_config['dsl_module'])

    program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, "list", "atom", 
        model.state_dim+model.action_dim, model.z_prog_dim,
        train_config['max_num_units'], train_config['min_num_units'], 
        train_config['max_depth'], train_config['penalty'])

    if "num_mc_samples" in train_config.keys():
        algorithm = MC_SAMPLING(num_mc_samples=train_config["num_mc_samples"])
    elif "frontier_capacity" in train_config.keys():
        algorithm = IDDFS_NEAR(frontier_capacity = train_config["frontier_capacity"])
    else:
        print("Implemented strategies include MC Sampling and NEAR.")
        raise NotImplementedError

    if hasattr(model, 'enc_prog_depth') and hasattr(model, 'enc_prog_cost'):
        curr_node = ProgramNode(
            model.enc_progs_learn[0], 0, None, model.enc_prog_depth, model.enc_prog_cost, 0)
    else:
        curr_node = program_graph.root_node

    # Don't want to duplicate fixed programs in model
    ignore_list = []
    for prog in model.enc_progs_fixed:
        assert program_graph.is_fully_symbolic(prog)
        ignore_list.append(prog.to_str())

    next_node = algorithm.update(
        curr_node, program_graph, batched_trainset, validset, train_config, device,
        ignore_list=ignore_list)

    model.set_enc_prog(next_node.program, enc_prog_type="learn", prog_ind=0)
    model.enc_prog_depth = next_node.depth
    model.enc_prog_cost = next_node.cost

    update_str = f"UPDATED PROGRAM: {model.enc_progs_learn[0].to_str(include_params=True)}\n"
    print(update_str)
    logger.experiment.log_text(update_str, metadata=logger_metadata)

    if program_graph.is_fully_symbolic(model.enc_progs_learn[0]):
        log_string = test_set_eval(model.enc_progs_learn[0], testset, "atom", num_labels, num_labels, evalfxn, device)
        logger.experiment.log_text(log_string, metadata=logger_metadata)

        delattr(model, 'enc_prog_depth')
        delattr(model, 'enc_prog_cost')

        return True
    else:
        return False

def extract_labels_from_model(model, dataloader):
    states_list, labels_list = [], []

    for batch in dataloader:
        states, actions = batch
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action, also assert batch_first=True
        batch_states = torch.cat([states[:,:-1], actions], dim=2)

        batch_labels_fixed, batch_labels_learn = model.get_labels(batch)
        assert len(batch_labels_learn) > 0

        states_list.append(batch_states)
        labels_list.append(batch_labels_learn[0]) # learn 1 program at a time

    return torch.cat(states_list, axis=0).numpy(), torch.cat(labels_list).numpy()
 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str,
                        required=False, default='',
                        help='path to all config files for experiments')
    parser.add_argument('--save_dir', type=str,
                        required=False, default='saved',
                        help='save directory for experiments from project directory')
    parser.add_argument('--exp_name', type=str,
                        required=False, default='',
                        help='experiment name (default will be config folder name)')
    parser.add_argument('-g', '--gpus', type=int,
                        required=False, default=0,
                        help='number of gpus to use')

    args = parser.parse_args()

    # Get JSON files
    config_dir = os.path.join(os.getcwd(), 'configs', args.config_dir)
    config_files = sorted([str(f) for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f))])
    assert len(config_files) > 0

    # Get experiment name
    exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_dir
    print('Config folder:\t {}'.format(exp_name))

    # Get save directory
    save_dir = os.path.join(os.getcwd(), args.save_dir, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'configs'))
    print('Save directory:\t {}'.format(save_dir))
        
    # Create master file
    master = {
        'start_time' : strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'experiment_name' : exp_name,
        'gpus' : args.gpus,
        'summaries' : {}
    }

    comet_logger = pl_loggers.CometLogger(
        save_dir = save_dir,
        workspace="comet-logs", 
        project_name="programmatic-generation" 
    )                
    comet_logger.experiment.add_tag(exp_name)

    main(args, config_files, save_dir, comet_logger)
