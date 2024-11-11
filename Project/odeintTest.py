import tensorflow as tf
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

# Define the parameters for the RLC circuit
R = 1.0  # Ohms
L = 1.0  # Henrys
C = 1.0  # Farads

I0 = 1.0
dI0_dt = 0.0

# Define the RLC ODE
def rlc_ode(t, y, alpha, omega_0):
    i, di_dt = y
    d2i_dt2 = -2 * alpha * di_dt - omega_0**2 * i
    return [di_dt, d2i_dt2]


# Define the input voltage as a function of time
def voltage(t):
    return 0

# Define the neural network model for current I(t)
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, t):
        x = self.dense1(t)
        x = self.dense2(x)
        return self.dense3(x)


# Define the custom loss function for the PINN
def pinn_loss(model, t, V, i0):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        I = model(t)
        dI_dt = tape.gradient(I, t)
        d2I_dt2 = tape.gradient(dI_dt, t)

    ode_loss = L * d2I_dt2 + R * dI_dt + (1 / C) * I + V(t)
    ic_loss = tf.square(I[0] - i0)
    return tf.reduce_mean(tf.square(ode_loss)) + ic_loss


# Define the training step
@tf.function
def train_step(model, t, V, i0, optimizer):
    with tf.GradientTape() as tape:
        loss = pinn_loss(model, t, V, i0)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Generate training data
t_values = np.linspace(0, 10, 1000).astype(np.float32).reshape(-1, 1)
t_train, t_test = train_test_split(t_values, test_size=0.2, random_state=42)

# Instantiate the model and optimizer
pinn_model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

# Train the model
loss_data = []
epochs = 2000
for epoch in range(epochs):
    loss = train_step(pinn_model, t_train, voltage, I0, optimizer)
    loss_data.append(loss.numpy())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Predict the current I(t) using the trained model
I_pred = pinn_model(t_values)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(t_values, I_pred, label='Predicted Current I(t)')
plt.xlabel('Time')
plt.ylabel('Current')
plt.title('Physics-Informed Neural Network for RLC Circuit')
plt.legend()
plt.show()

plt.plot(loss_data, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()