import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import qMq

# Load the original V1 data files
with open('log_obj_EKF_20241015143753.pkl', 'rb') as file:
    EKF_loaded_data = pickle.load(file)
with open('log_obj_UKF_20241023162946.pkl', 'rb') as file:
    UKF_loaded_data = pickle.load(file)
with open('log_obj_UKF_20241023162924.pkl', 'rb') as file:
    AI_UKF_loaded_data = pickle.load(file)
# with open('log_obj_EKF_20241023144552.pkl', 'rb') as file:
#     AI_EKF_loaded_data = pickle.load(file)

# Load the new V2 data files
with open('log_obj_UKF_20241111165454.pkl', 'rb') as file:
    V2_UKF_loaded_data = pickle.load(file)
with open('log_obj_UKF_20241111185410.pkl', 'rb') as file:
    V2_AI_UKF_loaded_data = pickle.load(file)

# Extract the state estimates and ground truth for V1 methods
# estimated_state_EKF = EKF_loaded_data.xhat_k
estimated_state_UKF = UKF_loaded_data.xhat_k
estimated_state_AI_UKF = AI_UKF_loaded_data.xhat_k
# estimated_state_AI_EKF = AI_EKF_loaded_data.xhat_k
true_state_V1 = UKF_loaded_data.x_k
time_V1 = EKF_loaded_data.time

# Extract the state estimates for V2 methods
estimated_state_V2_UKF = V2_UKF_loaded_data.xhat_k
estimated_state_V2_AI_UKF = V2_AI_UKF_loaded_data.xhat_k
true_state_V2 = V2_UKF_loaded_data.x_k
time_V2 = V2_UKF_loaded_data.time

del UKF_loaded_data, AI_UKF_loaded_data,
V2_UKF_loaded_data, V2_AI_UKF_loaded_data

estimated_state_V2_AI_UKF_ = np.array(estimated_state_V2_AI_UKF)
estimated_state_V2_UKF_ = np.array(estimated_state_V2_UKF)


# Convert list of tensors to numpy arrays for plotting (V1)
# estimated_state_EKF_np = [x.numpy()[:10] for x in estimated_state_EKF]
estimated_state_UKF_np = [x.numpy()[:10] for x in estimated_state_UKF]
# estimated_state_AI_EKF_np = [x.numpy()[:10] for x in estimated_state_AI_EKF]
estimated_state_AI_UKF_np = [x.numpy()[:10] for x in estimated_state_AI_UKF]
true_state_V1_np = [x.numpy()[:10] for x in true_state_V1]
time_V1_np = [(t - time_V1[0]) / 1e9 for t in time_V1]

#save estimated and true states for V1 as csv
np.savetxt('Deep_estimated_state_UKF.csv', np.array(estimated_state_AI_UKF_np), delimiter=',')
np.savetxt('Deep_true_state_UKF.csv', np.array(true_state_V1_np), delimiter=',')
np.savetxt('Deep_time.csv', np.array(time_V1_np), delimiter=',')


# Convert list of tensors to numpy arrays for plotting (V2)
estimated_state_V2_UKF_np = [x.numpy()[:10] for x in estimated_state_V2_UKF]
estimated_state_V2_AI_UKF_np = [x.numpy()[:10] for x in estimated_state_V2_AI_UKF]
true_state_V2_np = [x.numpy()[:10] for x in true_state_V2]
time_V2_np = [(t - time_V2[0]) / 1e9 for t in time_V2]

# Plot V1 State Trajectory Comparison
num_states_V1 = estimated_state_UKF_np[0].shape[0]
fig, axes = plt.subplots(num_states_V1, 1, figsize=(10, 5 * num_states_V1))
for state_idx in range(num_states_V1):
    ax = axes[state_idx] if num_states_V1 > 1 else axes
    # estimated_values_EKF = [state[state_idx] for state in estimated_state_EKF_np]
    estimated_values_UKF = [state[state_idx] for state in estimated_state_UKF_np]
    estimated_values_AI_UKF = [state[state_idx] for state in estimated_state_AI_UKF_np]
    # estimated_values_AI_EKF = [state[state_idx] for state in estimated_state_AI_EKF_np]
    true_values = [state[state_idx] for state in true_state_V1_np]

    ax.plot(time_V1_np, true_values, label='Ground Truth', linestyle='-')
    # ax.plot(time_V1_np, estimated_values_EKF, label='EKF', linestyle='--')
    ax.plot(time_V1_np, estimated_values_UKF, label='UKF', linestyle='-.')
    ax.plot(time_V1_np, estimated_values_AI_UKF, label='AI-UKF', linestyle=':')
    # ax.plot(time_V1_np, estimated_values_AI_EKF, label='AI-EKF', linestyle='-.')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('V1_state_trajectory_comparison.png' , dpi=600)
plt.close()

# Plot V2 State Trajectory Comparison
num_states_V2 = estimated_state_V2_UKF_np[0].shape[0]
fig, axes = plt.subplots(num_states_V2, 1, figsize=(10, 5 * num_states_V2))
for state_idx in range(num_states_V2):
    ax = axes[state_idx] if num_states_V2 > 1 else axes
    estimated_values_V2_UKF = [state[state_idx] for state in estimated_state_V2_UKF_np]
    estimated_values_V2_AI_UKF = [state[state_idx] for state in estimated_state_V2_AI_UKF_np]
    true_values = [state[state_idx] for state in true_state_V2_np]

    ax.plot(time_V2_np, true_values, label='Ground Truth', linestyle='-')
    ax.plot(time_V2_np, estimated_values_V2_UKF, label='V2 UKF', linestyle='--')
    ax.plot(time_V2_np, estimated_values_V2_AI_UKF, label='V2 AI-UKF', linestyle='-.')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('V2_state_trajectory_comparison.png', dpi=600)
plt.close()

# Quaternion error calculation for V1
# errors_quat_EKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_EKF_np, true_state_V1_np)]
errors_quat_UKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_UKF_np, true_state_V1_np)]
errors_quat_AI_UKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_AI_UKF_np, true_state_V1_np)]
# errors_quat_AI_EKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_AI_EKF_np, true_state_V1_np)]

# Quaternion error calculation for V2
errors_quat_V2_UKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_V2_UKF_np, true_state_V2_np)]
errors_quat_V2_AI_UKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4])) for est, true in zip(estimated_state_V2_AI_UKF_np, true_state_V2_np)]

# Save quaternion error plots separately for V1 and V2
plt.figure()
# plt.plot(time_V1_np, [e[0].item() for e in errors_quat_EKF], label='EKF')
plt.plot(time_V1_np, [e[0].item() for e in errors_quat_UKF], label='UKF')
plt.plot(time_V1_np, [e[0].item() for e in errors_quat_AI_UKF], label='AI-UKF')
# plt.plot(time_V1_np, [e[0].item() for e in errors_quat_AI_EKF], label='AI-EKF')
plt.legend()
plt.grid(True)
plt.savefig('V1_quaternion_error.png', dpi=600)
plt.close()

plt.figure()
plt.plot(time_V2_np, [e[0].item() for e in errors_quat_V2_UKF], label='V2 UKF')
plt.plot(time_V2_np, [e[0].item() for e in errors_quat_V2_AI_UKF], label='V2 AI-UKF')
plt.legend()
plt.grid(True)
plt.savefig('V2_quaternion_error.png', dpi=600)
plt.close()

def compute_mse(estimated_states, true_states):
    # Consider data points from the 50th onward
    estimated_states = estimated_states[50:]
    true_states = true_states[50:]

    # Separate quaternions (first 4 components), position (next 3), and velocity (last 3)
    quat_est = np.array([est[:4] for est in estimated_states])
    quat_true = np.array([true[:4] for true in true_states])
    pos_est = np.array([est[4:7] for est in estimated_states])
    pos_true = np.array([true[4:7] for true in true_states])
    vel_est = np.array([est[7:10] for est in estimated_states])
    vel_true = np.array([true[7:10] for true in true_states])

    # Calculate MSE for each component
    errors_quat = np.array([qMq(torch.tensor(est[:4]), torch.tensor(true[:4])).numpy() for est, true in zip(estimated_states, true_states)])
    mse_quat = np.mean(errors_quat ** 2)
    mse_pos = np.mean((pos_est - pos_true) ** 2)
    mse_vel = np.mean((vel_est - vel_true) ** 2)
    
    return mse_quat, mse_pos, mse_vel

# Calculate MSE for V1 methods
print("\nV1 MSE Errors:")
# mse_EKF = compute_mse(estimated_state_EKF, true_state_V1)
mse_UKF = compute_mse(estimated_state_UKF, true_state_V1)
mse_AI_UKF = compute_mse(estimated_state_AI_UKF, true_state_V1)
# mse_AI_EKF = compute_mse(estimated_state_AI_EKF, true_state_V1)
# print("EKF MSE:", mse_EKF)
print("UKF MSE:", mse_UKF)
print("AI-UKF MSE:", mse_AI_UKF)
# print("AI-EKF MSE:", mse_AI_EKF)

# Calculate MSE for V2 methods
print("\nV2 MSE Errors:")
mse_V2_UKF = compute_mse(estimated_state_V2_UKF, true_state_V2)
mse_V2_AI_UKF = compute_mse(estimated_state_V2_AI_UKF, true_state_V2)
print("V2 UKF MSE:", mse_V2_UKF)
print("V2 AI-UKF MSE:", mse_V2_AI_UKF)


import matplotlib.pyplot as plt
from numpy.linalg import norm
import torch

# Create a new figure and tiled layout
fig = plt.figure(figsize=(15, 8))
# t = plt.GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.4)
t = plt.GridSpec(3, 4, figure=fig)

# Calculate position errors
PGrdtruth = np.array([x[4:7] for x in true_state_V1]).T
Pestimate = np.array([x[4:7] for x in estimated_state_AI_UKF]).T
errors = np.linalg.norm((PGrdtruth - Pestimate), axis=0)

# Calculate quaternion axes
q_estimate = np.array([x[:4] for x in estimated_state_AI_UKF])
# indices = np.round(np.linspace(0, PGrdtruth.shape[1] - 1, 3)).astype(int)
indices = [int((PGrdtruth.shape[1] - 1)/3) , 0 , int(2*(PGrdtruth.shape[1] - 1)/4)]
axis_length = 0.35

# Plot ground truth trajectory with AI UKF estimate
ax_main = fig.add_subplot(t[:, :3], projection='3d')
ax_main.plot(PGrdtruth[0, :], PGrdtruth[1, :], PGrdtruth[2, :], '--', color='black', linewidth=1, label='Ground Truth')
ax_main.set_xlabel('$X \, [\mathrm{m}]$', fontsize=14, labelpad=12)
ax_main.set_ylabel('$Y \, [\mathrm{m}]$', fontsize=14, labelpad=12)
ax_main.set_zlabel('$Z \, [\mathrm{m}]$', fontsize=14, labelpad=12)
ax_main.set_title('MAV Trajectory', fontsize=16, pad=20)
ax_main.grid(True)

# Plot quaternion axes at selected points
for i in indices:
    q_ = q_estimate[i]
    R = torch.tensor([
        [1 - 2*(q_[2]**2 + q_[3]**2), 2*(q_[1]*q_[2] - q_[3]*q_[0]), 2*(q_[1]*q_[3] + q_[2]*q_[0])],
        [2*(q_[1]*q_[2] + q_[3]*q_[0]), 1 - 2*(q_[1]**2 + q_[3]**2), 2*(q_[2]*q_[3] - q_[1]*q_[0])],
        [2*(q_[1]*q_[3] - q_[2]*q_[0]), 2*(q_[2]*q_[3] + q_[1]*q_[0]), 1 - 2*(q_[1]**2 + q_[2]**2)]
    ])
    origin = PGrdtruth[:, i]

    # Plot axes with different colors for X, Y, Z
    for j, color in enumerate(['r', 'g', 'b']):
        end_point = origin + axis_length * R[:, j].numpy()
        ax_main.plot(
            [origin[0], end_point[0]], 
            [origin[1], end_point[1]], 
            [origin[2], end_point[2]], 
            f'{color}--', linewidth=2
        )

# Set limits for better visualization if necessary (optional)
ax_main.set_xlim([min(PGrdtruth[0, :]), max(PGrdtruth[0, :])])
ax_main.set_ylim([min(PGrdtruth[1, :]), max(PGrdtruth[1, :])])
ax_main.set_zlim([min(PGrdtruth[2, :]), max(PGrdtruth[2, :])])
# Add legend for clarity
# ax_main.legend(loc='upper right', fontsize=12)
# Errors for rotation, position, and velocity
rot_error = [norm(qMq(torch.tensor(est[:4]), torch.tensor(true[:4])).clone().detach().numpy()) for est, true in zip(estimated_state_AI_UKF, true_state_V1)]
pos_error = norm(PGrdtruth - Pestimate, axis=0)
vel_error = norm(np.array([x[7:10] for x in true_state_V1]).T - np.array([x[7:10] for x in estimated_state_AI_UKF]).T, axis=0)

# Plot rotation error
ax_rot = fig.add_subplot(t[0, 3])
ax_rot.plot(time_V1_np, rot_error, 'b-', linewidth=2)
ax_rot.set_title('Rotation Error', fontsize=14)
ax_rot.set_ylabel('$\|r_{e,k}\| \, [\mathrm{rad}]$', fontsize=14)
ax_rot.set_xlim([time_V1_np[0], time_V1_np[-1]])
ax_rot.set_ylim([0, 1.1 * max(rot_error)])
ax_rot.grid(True)
ax_rot.tick_params(axis='both', which='major', labelsize=12)

# Plot position error
ax_pos = fig.add_subplot(t[1, 3])
ax_pos.plot(time_V1_np, pos_error, 'b-', linewidth=2)
ax_pos.set_title('Position Error', fontsize=14)
ax_pos.set_ylabel('$\|p_{e,k}\| \, [\mathrm{m}]$', fontsize=14)
ax_pos.set_xlim([time_V1_np[0], time_V1_np[-1]])
ax_pos.set_ylim([0, 1.1 * max(pos_error)])
ax_pos.grid(True)
ax_pos.tick_params(axis='both', which='major', labelsize=12)

# Plot velocity error
ax_vel = fig.add_subplot(t[2, 3])
ax_vel.plot(time_V1_np, vel_error, 'b-', linewidth=2)
ax_vel.set_title('Velocity Error', fontsize=14)
ax_vel.set_ylabel('$\|v_{e,k}\| \, [\mathrm{m/s}]$', fontsize=14)
ax_vel.set_xlabel('Time $[\mathrm{s}]$', fontsize=14)
ax_vel.set_xlim([time_V1_np[0], time_V1_np[-1]])
ax_vel.set_ylim([0, 1.1 * max(vel_error)])
ax_vel.grid(True)
ax_vel.tick_params(axis='both', which='major', labelsize=12)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('trajectory_and_errors_v1_ai_ukf.png', dpi=600)
plt.show()
plt.close()

# Create a 3x3 grid to plot the errors for V1 AI-UKF
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
# fig.suptitle('Orientation, Position, and Velocity Errors for V1 AI-UKF', fontsize=20)

# Extract and calculate the errors for AI-UKF (V1)
orientation_errors_AI_UKF = [qMq(torch.tensor(est[:4]), torch.tensor(true[:4]))[0].item() for est, true in zip(estimated_state_AI_UKF_np, true_state_V1_np)]
position_errors_AI_UKF = np.linalg.norm(np.array([est[4:7] - true[4:7] for est, true in zip(estimated_state_AI_UKF_np, true_state_V1_np)]), axis=1)
velocity_errors_AI_UKF = np.linalg.norm(np.array([est[7:10] - true[7:10] for est, true in zip(estimated_state_AI_UKF_np, true_state_V1_np)]), axis=1)

# Plot orientation errors in the first column
for i in range(3):
    axes[i, 0].plot(time_V1_np, orientation_errors_AI_UKF, 'b-', linewidth=2)
    axes[0, 0].set_title('Orientation Error', fontsize=20)
    axes[2, 0].set_xlabel('Time [s]', fontsize=20)
    axes[i, 0].set_ylabel(f'$r_{{e,k,{i+1}}} \, [\\text{{m.rad}}]$', fontsize=20)
    axes[i, 0].set_xlim([time_V1_np[0]-1, time_V1_np[-1]])
    axes[i, 0].set_ylim([1.1 * min(orientation_errors_AI_UKF), 1.1 * max(orientation_errors_AI_UKF)])
    axes[i, 0].grid(True)
    axes[i, 0].tick_params(axis='both', which='major', labelsize=12)

# Plot position errors in the second column
for i in range(3):
    axes[i, 1].plot(time_V1_np, position_errors_AI_UKF, 'b-', linewidth=2)
    axes[0, 1].set_title('Position Error', fontsize=20)
    axes[2, 1].set_xlabel('Time [s]', fontsize=20)
    axes[i, 1].set_ylabel(f'$p_{{e,k,{i+1}}} \, [\\text{{m}}]$', fontsize=20)
    axes[i, 1].set_xlim([time_V1_np[0]-1, time_V1_np[-1]])
    axes[i, 1].set_ylim([0, 1.1 * max(position_errors_AI_UKF)])
    axes[i, 1].grid(True)
    axes[i, 1].tick_params(axis='both', which='major', labelsize=12)

# Plot velocity errors in the third column
for i in range(3):
    axes[i, 2].plot(time_V1_np, velocity_errors_AI_UKF, 'b-', linewidth=2)
    axes[0, 2].set_title('Velocity Error', fontsize=20)
    axes[2, 2].set_xlabel('Time [s]', fontsize=20)
    axes[i, 2].set_ylabel(f'$v_{{e,k,{i+1}}} \, [\\text{{m/s}}]$', fontsize=20)
    axes[i, 2].set_xlim([time_V1_np[0]-1, time_V1_np[-1]])
    axes[i, 2].set_ylim([0, 1.1 * max(velocity_errors_AI_UKF)])
    axes[i, 2].grid(True)
    axes[i, 2].tick_params(axis='both', which='major', labelsize=12)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('v1_ai_ukf_errors_orientation_position_velocity.png', dpi=600)
plt.show()


from AI_model_utils import estimation_loss
loss_fn = estimation_loss()
# Calculate and print the loss for V1 methods
print("\nV1 Losses:")
# loss_EKF = loss_fn(torch.stack(estimated_state_EKF), torch.stack(true_state_V1))
loss_UKF = loss_fn(torch.stack(estimated_state_UKF), torch.stack(true_state_V1))
loss_AI_UKF = loss_fn(torch.stack(estimated_state_AI_UKF), torch.stack(true_state_V1))
# loss_AI_EKF = loss_fn(torch.stack(estimated_state_AI_EKF), torch.stack(true_state_V1))
# print("EKF Loss:", loss_EKF.item())
print("UKF Loss:", loss_UKF.item())
print("AI-UKF Loss:", loss_AI_UKF.item())
# print("AI-EKF Loss:", loss_AI_EKF.item())

# Calculate and print the loss for V2 methods
print("\nV2 Losses:")
loss_V2_UKF = loss_fn(torch.stack(estimated_state_V2_UKF), torch.stack(true_state_V2))
loss_V2_AI_UKF = loss_fn(torch.stack(estimated_state_V2_AI_UKF), torch.stack(true_state_V2))
print("V2 UKF Loss:", loss_V2_UKF.item())
print("V2 AI-UKF Loss:", loss_V2_AI_UKF.item())


