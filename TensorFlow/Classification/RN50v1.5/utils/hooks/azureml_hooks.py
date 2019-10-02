#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np
import tensorflow as tf

__all__ = ['AzureMLLoggingHook']


class AzureMLLoggingHook(tf.train.SessionRunHook):

    def __init__(
        self, global_batch_size, log_every=10
    ):
        from azureml.core import Run
        self.run = Run.get_context()

        self.log_every = log_every
        self.global_batch_size = global_batch_size
        self.latest_step_seen = -1

    def before_run(self, run_context):
        run_args = tf.train.SessionRunArgs(
            fetches=[
                tf.train.get_global_step(), 'cross_entropy_loss_ref:0', 'l2_loss_ref:0', 'total_loss_ref:0',
                'learning_rate_ref:0'
            ]
        )
        self.t0 = time.time()

        return run_args

    def after_run(self, run_context, run_values):
        global_step, cross_entropy, l2_loss, total_loss, learning_rate = run_values.results
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time
        
        if int(global_step) > self.latest_step_seen and int(global_step) % self.log_every == 0:
            self.run.log_row(name='Loss curve', step=int(global_step), loss=float(cross_entropy))
            self.run.log_row(name='Learning rate schedule', step=int(global_step), lr=float(learning_rate))
            self.run.log_row(name='Throughput', step=int(global_step), ips=float(ips))
        
        self.latest_step_seen = int(global_step)

    def end(self, session):
        pass
