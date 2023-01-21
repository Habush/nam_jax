from tqdm import tqdm
from models import *
from utils import *
from tqdm import tqdm

from models import *
from utils import *


def cross_entropy_loss(preds, target):
    loss = optax.sigmoid_binary_cross_entropy(preds, target)
    return loss

def mse_loss(preds, target):
    return jnp.mean((preds - target)**2)

def reg_eval(params, model, rng, x, y):
    preds, _ = model.apply(params, rng, x, False)
    return jnp.sqrt(mse_loss(preds, y))

def classification_eval(params, model, rng, x, y):
    preds, _ = model.apply(params, rng, x, False)
    return cross_entropy_loss(preds, y)



def train_loop(seed, x_train, x_val, x_test, y_train, y_val, y_test,
               epochs, batch_size, output_reg, dropout, schedule_fn,
               l2_reg=0.0, feature_dropout=0.0,  num_basis_functions = 1000,
               units_multiplier = 2, activation = 'relu',
               eval_epoch=50, shallow=True, regression=True):

    key = jax.random.PRNGKey(seed)

    if regression:
        loss_fn = mse_loss
        eval_fn = reg_eval
    else:
        loss_fn = cross_entropy_loss
        eval_fn = classification_eval

    torch.manual_seed(seed)
    data_loader = NumpyLoader(NumpyData(x_train, y_train), batch_size=batch_size, shuffle=False)

    num_unique_vals = [
        len(jnp.unique(x_train[:, i])) for i in range(x_train.shape[1])
    ]
    num_units = [
        min(num_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]
    model = NAM(num_inputs=x_train.shape[-1] ,num_units=num_units,
                step_size_fn=schedule_fn,
                dropout=jnp.float32(dropout),
                feature_dropout=jnp.float32(feature_dropout),
                activation=activation,
                shallow=shallow,
                output_reg=output_reg,
                l2_reg=l2_reg, loss_fn=loss_fn)

    state = model.init(key, next(iter(data_loader))[0])

    val_losses = []
    for epoch in tqdm(range(epochs)):
        for x, y in data_loader:
            _, key = jax.random.split(key, 2)
            state, _ = model.update(state, key, x, y)

        if epoch % eval_epoch == 0:
            loss = eval_fn(state.params, model, key, x_val, y_val)
            val_losses.append(loss)
            print(f"Epoch: {epoch}, loss: {loss}")

    test_loss = eval_fn(state.params, model, key, x_test, y_test)

    return model, state, val_losses, test_loss