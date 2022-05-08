# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""

import functools

from absl import app
from absl import flags
import tensorflow_federated as tff

from drive import fed_avg_schedule
from drive.drive_utils import drive_roundtrip_builder, _create_drive_fn
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

import tensorflow as tf

from absl import logging

logging.info(f"!tf.config.list_physical_devices('GPU')= {tf.config.list_physical_devices('GPU')}")

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
    'max_elements_per_client', None, 'Maximum number of '
                                     'elements for each training client. If set to None, all '
                                     'available examples are used.')

  # Training loop configuration
  flags.DEFINE_string(
    'experiment_name', None, 'The name of this experiment. Will be append to '
                             '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', 'cs_storage/amitport/extended-drive/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
    'rounds_per_eval', 1,
    'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
    'num_validation_examples', -1, 'The number of validation'
                                   'examples to use. If set to -1, all available examples '
                                   'are used.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as compression_flags:
  flags.DEFINE_integer('num_bits', 4, 'Number of bits for quantization.')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task flags
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


class Drive:
  def __init__(self, bits, hadamard_repeats):
    self._bits = bits
    self._hadamard_repeats = hadamard_repeats
    self._roundtrip = drive_roundtrip_builder(bits=bits, hadamard_repeats=hadamard_repeats)

  def roundtrip(self, x):
    return self._roundtrip(x)

  def __str__(self):
    return f'Compression = Drive {self._bits=} {self._hadamard_repeats=}'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  compression_dict = utils_impl.lookup_flag_values(compression_flags)

  train_client_spec = tff.simulation.baselines.ClientSpec(
    num_epochs=FLAGS.client_epochs_per_round,
    batch_size=FLAGS.client_batch_size,
    max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  # compression = Drive(bits=1, hadamard_repeats=1)
  # print(compression)
  bits = compression_dict['num_bits']
  create_compress_roundtrip_fn = functools.partial(_create_drive_fn, bits=bits, hadamard_repeats=1)
  logging.info(f"DRIVE bits={bits}, hadamard_repeats=1")

  def _create_noop_fn(value_type):
    @tff.tf_computation(value_type)
    def noop_fn(record):
      return record

    return noop_fn

  if bits == 32:
    logging.info('NOOP!')
    create_compress_roundtrip_fn = _create_noop_fn

  iterative_process = fed_avg_schedule.build_fed_avg_process(
    create_compress_roundtrip_fn=create_compress_roundtrip_fn,
    model_fn=task.model_fn,
    client_optimizer_fn=client_optimizer_fn,
    client_lr=client_lr_schedule,
    server_optimizer_fn=server_optimizer_fn,
    server_lr=server_lr_schedule,
    client_weight_fn=None)
  train_data = task.datasets.train_data.preprocess(
    task.datasets.train_preprocess_fn)
  training_process = (
    tff.simulation.compose_dataset_computation_with_iterative_process(
      train_data.dataset_computation, iterative_process))

  training_selection_fn = functools.partial(
    tff.simulation.build_uniform_sampling_fn(
      train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
    size=FLAGS.clients_per_round)

  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(FLAGS.num_validation_examples)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)
  evaluation_selection_fn = lambda round_num: [validation_data]

  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
    FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparam_flags()
  state = tff.simulation.run_training_process(
    training_process=training_process,
    training_selection_fn=training_selection_fn,
    total_rounds=FLAGS.total_rounds,
    evaluation_fn=evaluation_fn,
    evaluation_selection_fn=evaluation_selection_fn,
    rounds_per_evaluation=FLAGS.rounds_per_eval,
    program_state_manager=program_state_manager,
    rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
    metrics_managers=metrics_managers)

  test_metrics = federated_eval(state.model, [test_data])
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
