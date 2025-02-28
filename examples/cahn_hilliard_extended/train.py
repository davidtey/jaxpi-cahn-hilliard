import os
import time

import jax
import jax.numpy as jnp
from jax import vmap, lax, grad
from jax.tree_util import tree_map
from jax.lib import xla_bridge

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import UniformSampler, TimeSpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

def train_init_condition(config, workdir, model, u_ref):
    logger = Logger()

    evaluator = models.CHEEvaluator(config, model)

    print("Waiting for JIT...")
    start_time = time.time()
    print(f"Device: {xla_bridge.get_backend().platform}")

    print(f"Training neural network approximation of initial condition...")

    for step in range(config.training.ics_warmup_max_steps):
        # train on the initial condition
        grads = grad(model.ics_warmup_loss)(model.state.params, state.weights, batch)
        grads = lax.pmean(grads, "batch")
        model.state = model.state.apply_gradients(grads=grads)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
                
                if log_dict['ics_loss'] < config.training.ics_warmup_error_break:
                    break


def train_one_window(config, workdir, model, samplers, u_ref, idx):
    logger = Logger()

    evaluator = models.CHEEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    print("Waiting for JIT...")
    start_time = time.time()
    print(f"Device: {xla_bridge.get_backend().platform}")
    for step in range(config.training.max_steps):
        # Sample mini-batch
        batch = {}
        # for key, sampler in samplers.items():
        #     batch[key] = next(sampler)
            
        batch['res'] = next(samplers['res'])
            
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time
                
                if log_dict['ics_loss'] < config.training.ics_error_break and log_dict['res_loss'] < config.training.res_error_break:
                    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                    save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
                    break

        # Save model checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Get the reference solution
    u_ref, t_star, x_star = get_dataset()

    u0 = u_ref[0, :]  # initial condition of the first time window

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Define the time and space domain
    t0 = t[0]
    t1 = t[-1] * (1 + 0.01)  # cover the end point of each time window
    x0 = x_star[0]
    x1 = x_star[-1]
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.res_batch_size))
    
    # Initialize the boundary condition sampler
    # bc_sampler = iter(TimeSpaceSampler(jnp.array([t0, t1]), jnp.array([[x0], [x1]]), config.training.boundary_batch_size))
    
    samplers = {}
    samplers['res'] = res_sampler
    # samplers['bc'] = bc_sampler

    # Init condition warmup
    model = models.CHE(config, u0, t, x_star)
    model = train_init_condition(config, workdir, model, u_ref)
    u0 = vmap(model.u_net, (None, None, 0))(model.state.params, t_star[num_time_steps], x_star)

    for idx in range(config.training.num_time_windows):
        print("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Initialize the model
        if idx > 0:
            model = models.CHE(config, u0, t, x_star)

        # Training the current time window
        model = train_one_window(config, workdir, model, samplers, u, idx)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0))(
                params, t_star[num_time_steps], x_star
            )

            del model, state, params
