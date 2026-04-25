import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 1. HARDWARE SIMULATION
puf = XORArbiterPUF(n=64, k=4, seed=1, noisiness=0.05)
n_crps = 2000
challenges = random_inputs(n=64, N=n_crps, seed=42)
responses = puf.eval(challenges)

# 2. FEATURE EXTRACTION
def get_features(data):
    return [np.mean(data), np.std(data)]

session_size = 50
X_train = [get_features(responses[i:i+session_size]) for i in range(0, len(responses), session_size)]

# 3. AI DETECTION (Isolation Forest)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)

# 4. ATTACK SIMULATION
attack_responses = np.random.choice([1, -1], size=1000) 
X_attack = [get_features(attack_responses[i:i+session_size]) for i in range(0, len(attack_responses), session_size)]

# 5. RESULTS & PLOT
results = model.predict(X_attack)
detected = np.count_nonzero(results == -1)
print(f"Attack Sessions Detected: {detected} out of {len(X_attack)}")

plt.scatter([f[0] for f in X_train], [f[1] for f in X_train], color='cyan', label='Normal')
plt.scatter([f[0] for f in X_attack], [f[1] for f in X_attack], color='red', marker='x', label='Attack')
plt.legend()
plt.title("PUF Security: Behavioral Anomaly Detection")
plt.show()


