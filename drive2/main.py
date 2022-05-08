import functools

from absl import app, logging
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_model_optimization.python.core.internal import \
  tensor_encoding as te

from drive.random_p import RandomPEncodingStage
from drive2.encoded import DriveSignStage
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils


with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

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
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
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

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


# AGGR_FLAGS = collections.OrderedDict(
#   sketch_sgd=sketch_sgd_flags,
#   uniform_quantization=uniform_quantization_flags,
#   hadamard_quantization=hadamard_quantization_flags,
#   qsgd=qsgd_flags,
#   adaq=adaq_flags,
#   sign_sgd=sign_sgd_flags,
#   topk=topk_flags,
#   threshold=threshold_flags,
#   sketch=sketch_flags,
#   dgc=dgc_flags,
#   randomk=randomk_flags,
#   unbiased_srb=unbiased_srb_flags,
#   hadamard_srb=hadamard_srb_flags,
# )

# AGGR_FLAG_PREFIXES = dict(zip(_SUPPORTED_AGGR, _SUPPORTED_AGGR))


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

  # aggr_name = FLAGS.aggr
  # if aggr_name in AGGR_FLAGS:
  #   aggr_hparam_dict = utils_impl.lookup_flag_values(AGGR_FLAGS[aggr_name])
  #   hparam_dict.update(aggr_hparam_dict)
  # hparam_dict.update([('aggr', aggr_name)])
  #
  # hparam_dict.update(utils_impl.lookup_flag_values(encoder_flags))

  # return hparam_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  def drive_encoder():
    return te.core.core_encoder.EncoderComposer(
      te.stages.BitpackingEncodingStage(1)
    ).add_parent(
      RandomPEncodingStage(p=0.1),
      RandomPEncodingStage.ENCODED_VALUES_KEY
    ).add_parent(
      DriveSignStage(),
      DriveSignStage.ENCODED_VALUES_KEY
    ).add_parent(
      te.stages.HadamardEncodingStage(),
      te.stages.HadamardEncodingStage.ENCODED_VALUES_KEY
    ).add_parent(
      te.stages.FlattenEncodingStage(),
      te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY
    ).make()

  def mean_encoder_fn(tensor_spec):
    """Function for building a GatherEncoder."""
    spec = tf.TensorSpec(tensor_spec.shape, tensor_spec.dtype)
    if tensor_spec.shape.num_elements() > 10000:
      return te.encoders.as_gather_encoder(drive_encoder(), spec)
    else:
      return te.encoders.as_gather_encoder(te.encoders.identity(), spec)

  mean_factory = tff.aggregators.MeanFactory(
    tff.aggregators.EncodedSumFactory(mean_encoder_fn)
  )

  logging.info('P=0.1!!')
  iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=task.model_fn,
    client_optimizer_fn=client_optimizer_fn,
    server_optimizer_fn=server_optimizer_fn,
    model_update_aggregation_factory=mean_factory,
  )
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
