import tensorflow as tf
import tensorflow_federated as tff


class MinMaxNormalizationFactory(tff.aggregators.UnweightedAggregationFactory):
    def __init__(self,
                 inner_agg_factory: tff.aggregators.UnweightedAggregationFactory):
        self._inner_agg_factory = inner_agg_factory

    def create(self, value_type):
        if not (tff.types.is_structure_of_floats(value_type) or
                (value_type.is_tensor() and value_type.dtype == tf.float32)):
            raise ValueError("Expect value_type to be float tensor or structure of "
                             f"float tensors, found {value_type}.")

        value_scalar_type = tff.TensorType(value_type.dtype)

        @tff.tf_computation(value_type, value_scalar_type, value_scalar_type)
        def normalize(value, client_min, client_max):
            @tf.function
            def transform(tensor):
                tensor = tf.math.divide_no_nan(tensor - client_min, client_max - client_min)
                return tensor

            return tf.nest.map_structure(transform, value)

        inner_agg_process = self._inner_agg_factory.create(value_type)

        @tff.tf_computation(value_type, value_scalar_type, value_scalar_type)
        def denormalize(value, client_min, client_max):
            @tf.function
            def transform(tensor):
                tensor = tensor * (client_max - client_min) + client_min
                return tensor

            return tf.nest.map_structure(transform, value)

        @tff.federated_computation()
        def init_fn():
            inner_state = inner_agg_process.initialize()
            return inner_state

        @tff.federated_computation(init_fn.type_signature.result,
                                   tff.type_at_clients(value_type))
        def next_fn(inner_state, value):
            client_min = tff.federated_map(tff.tf_computation(lambda x: tf.nest.map_structure(tf.math.reduce_min, x)),
                                           value)
            client_max = tff.federated_map(tff.tf_computation(lambda x: tf.nest.map_structure(tf.math.reduce_max, x)),
                                           value)

            normalized_value = tff.federated_map(normalize, (value, client_min, client_max))

            inner_agg_output = inner_agg_process.next(inner_state, normalized_value)

            denormalized_value = tff.federated_map(denormalize, (
                normalized_value, client_min, client_max))

            new_state = inner_agg_output.state

            measurements = inner_agg_output.measurements

            return tff.templates.MeasuredProcessOutput(
                state=new_state, result=inner_agg_output.result, measurements=measurements)

        return tff.templates.AggregationProcess(init_fn, next_fn)
