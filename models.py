import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import NamedTuple

class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState

def exu(x, weight, bias):
    """ExU hidden unit modification."""
    return (x - bias) @ jnp.exp(weight)


# Activation Functions
def relu(x, weight, bias):
    """ReLU activation."""
    return jax.nn.relu((x - bias) @ weight)


def relu_n(x, n = 6):
    """ReLU activation clipped at n."""
    return jnp.clip(x, 0, n)


class FeatureNet(hk.Module):
    """Neural Network model for each individual feature.

    Attributes:
      hidden_layers: A list containing hidden layers. The first layer is an
        `ActivationLayer` containing `num_units` neurons with specified
        `activation`. If `shallow` is False, then it additionally contains 2
        tf.keras.layers.Dense ReLU layers with 64, 32 hidden units respectively.
      linear: Fully connected layer.
    """

    def __init__(self,
                 input_shape,
                 num_units,
                 dropout = 0.5,
                 shallow = True,
                 feature_num = 0,
                 activation = 'exu',
                 name=None):
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          shallow: If True, then a shallow network with a single hidden layer is
            created, otherwise, a network with 3 hidden layers is created.
          feature_num: Feature Index used for naming the hidden layers.
          activation: Activation and type of hidden unit(ExUs/Standard) used in the
            first hidden layer.
        """


        if name is not None:
            super().__init__(name)
        else:
            super().__init__(f"f_{feature_num}")
        self._input_shape = input_shape
        self._num_units = num_units
        self._dropout = dropout
        self._feature_num = feature_num
        self._shallow = shallow

        self._activation = activation

        if activation == "exu":
            self._act_fn = exu
            self._initializer = hk.initializers.TruncatedNormal(mean=4, stddev=0.5)

        else:
            self._act_fn = relu
            self._initializer = hk.initializers.UniformScaling()


    def __call__(self, x, is_training):
        key = hk.next_rng_key()
        k1, k2 = jax.random.split(key, 2)
        w = hk.get_parameter("w", [self._input_shape, self._num_units], init=self._initializer)
        c = hk.get_parameter("c", [self._input_shape], init=hk.initializers.TruncatedNormal(stddev=0.5))

        # x = relu_n(self._act_fn(x, w, b), 1)
        x = self._act_fn(x, w, c)
        if not self._shallow:
            x = hk.Linear(64, w_init=hk.initializers.UniformScaling())(x)
            x = jax.nn.relu(x)
            if is_training:
                x = hk.dropout(k1, self._dropout, x)
            x = hk.Linear(32, w_init=hk.initializers.UniformScaling())(x)
            if is_training:
                x = hk.dropout(k2, self._dropout, x)
            x = jax.nn.relu(x)


        x = hk.Linear(1, with_bias=False, w_init=hk.initializers.UniformScaling())(x)

        return x

    # def __call__(self, params, key, x, is_training):
    #     return self._forward.apply(params, key, x, is_training).ravel()

class NAM:
    """Neural additive model.

    Attributes:
      feature_nns: List of FeatureNN, one per input feature.
    """
    def __init__(self, *,
                 num_inputs,
                 num_units,
                 step_size_fn,
                 loss_fn,
                 shallow = True,
                 feature_dropout = 0.0,
                 dropout = 0.0,
                 output_reg = 0.0,
                 l2_reg = 0.0,
                 activation="exu"):

        """Initializes NAM hyperparameters.

        Args:
          num_units: Number of hidden units in first layer of each feature net.
          trainable: Whether the NAM parameters are trainable or not.
          shallow: If True, then shallow feature nets with a single hidden layer are
            created, otherwise, feature nets with 3 hidden layers are created.
          feature_dropout: Coefficient for dropping out entire Feature NNs.
          dropout: Coefficient for dropout within each Feature NNs.
          **kwargs: Arbitrary keyword arguments. Used for passing the `activation`
            function as well as the `name_scope`.
        """

        self._num_inputs = num_inputs
        self._shallow = shallow
        self._feature_dropout = feature_dropout
        self._dropout = dropout
        self.loss_fn = loss_fn
        self._activation = activation
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]

        self.optimiser = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(step_size_fn), optax.scale(-1.0))

        self._forward = hk.transform(self._forward_fn)
        self.update = jax.jit(self.update)

    def init(self, key, x):
        params = self._forward.init(key, x, True)
        opt_state = self.optimiser.init(params)
        return TrainingState(params, params, opt_state)

    def apply(self, params, key, x, is_training=True):
        pred, per_feat_pred =  self._forward.apply(params, key, x, is_training)
        return pred, per_feat_pred

    def update(self, state, key, x, y):
        grads, per_feat_pred = jax.grad(self.loss, has_aux=True)(state.params, key, x, y, True)
        updates, opt_state = self.optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        avg_params = optax.incremental_update(params, state.avg_params, step_size=1e-3)
        return TrainingState(params, avg_params, opt_state), per_feat_pred


    def loss(self, params, key, x, y, is_training):
        preds, per_feat_pred = self.apply(params, key, x, is_training)
        loss = self.loss_fn(preds, y)
        reg_loss = 0.0

        per_feature_norm = [jnp.mean(jnp.square(outputs)) for outputs in per_feat_pred]
        per_feature_norm = sum(per_feature_norm) / len(per_feature_norm)
        l2_features = self.output_reg*per_feature_norm
        reg_loss += l2_features

        l2_per_coef = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)) / len(per_feat_pred)

        reg_loss += self.l2_reg*l2_per_coef

        return loss + reg_loss, per_feat_pred

    def _forward_fn(self, x, is_training):
        key = hk.next_rng_key()
        x = jnp.split(x, self._num_inputs, axis=-1)

        per_feat_out = []
        for i in range(self._num_inputs):
            per_feat_out.append(FeatureNet(1, self._num_units[i], self._dropout, self._shallow, i, self._activation)(x[i], is_training).squeeze())

        if is_training:
            out = sum(jax.tree_util.tree_map(lambda l: hk.dropout(key, self._feature_dropout, l), per_feat_out))
        else:
            out = sum(per_feat_out)

        b = hk.get_parameter("b", [1,], init=hk.initializers.Constant(0))

        return out + b, per_feat_out