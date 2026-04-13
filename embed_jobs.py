import pickle
import numpy as np
from transformers import BertModel
import matplotlib.pyplot as plt

def load_data(file_path):
    data = []
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_n_closest_vectors(mat, vec, n):
    dists = np.linalg.norm(mat - vec, axis=1)
    closest = [[None, None] for x in range(n)]
    for x in range(len(dists)):
        dist = dists[x]
        for i in range(n):
            if (closest[i][1] is None) or (dist < closest[i][1]):
                closest.insert(i, [x, dist])
                closest.pop()
                break
    return closest

def get_n_furthest_vectors(mat, vec, n):
    dists = np.linalg.norm(mat - vec, axis=1)
    furthest = [[None, None] for x in range(n)]
    for x in range(len(dists)):
        dist = dists[x]
        for i in range(n):
            if furthest[i][1] is None or dist > furthest[i][1]:
                furthest.insert(i, [x, dist])
                furthest.pop()
                break
    return furthest

#################################################################

data_path = "jobs_data/"
desc_data_path = data_path + "split_desc_data.pkl"
data_path = data_path + "split_data.pkl"

desc_data = load_data(desc_data_path)
data = load_data(data_path)

desc_training = desc_data["training"]

data_pts = len(desc_training)
print()
print(f"Number of training data points: {data_pts}")
print()

DEVICE = 'cuda'
model = BertModel.from_pretrained('bert-base-uncased').to(device=DEVICE)

all_embeddings = []
for x in range(data_pts):
    embedding = model(input_ids=desc_training[x]["input_ids"].to(device=DEVICE), attention_mask=desc_training[x]["attention_mask"].to(device=DEVICE)) # FIX This training data is not in the correct format, it also takes 3min to process
    all_embeddings.append(embedding["pooler_output"].cpu().detach().numpy())

embeddings_memory = all_embeddings[0].size * all_embeddings[0].itemsize
all_embeddings_memory = embeddings_memory * len(all_embeddings)

all_data = np.asarray(all_embeddings)
all_data = all_data.squeeze()

mean_all_data = np.mean(all_data, axis=0)
std_all_data = np.std(all_data, axis=0)

number = 3
data_mean_closest = get_n_closest_vectors(all_data, mean_all_data, number)
data_mean_furthest = get_n_furthest_vectors(all_data, mean_all_data, number)

print()
print("closest to mean: ", data_mean_closest)
for x in range(number):
    print()
    print(data["training"][data_mean_closest[x][0]])

print()
print("furthest to mean: ", data_mean_furthest)
for x in range(number):
    print()
    print(data["training"][data_mean_furthest[x][0]])
print()

all_data = (all_data - mean_all_data) / std_all_data
U, S, V = np.linalg.svd(all_data)

num_eigvals = S.shape[0]
proj = np.dot(U[:,:num_eigvals] * S, V[:num_eigvals,:2])

mean_proj = np.mean(proj, axis=0)
print()
print("closest point projections:")
closest_proj_vec = []
for x in range(number):
    closest_proj = proj[data_mean_closest[x][0]]
    closest_proj_vec.append(closest_proj.tolist())
closest_proj_vec = np.asarray(closest_proj_vec)
print(closest_proj_vec)

print()
print("furthest point projections:")
furthest_proj_vec = []
for x in range(number):
    furthest_proj = proj[data_mean_furthest[x][0]]
    furthest_proj_vec.append(furthest_proj.tolist())
furthest_proj_vec = np.asarray(furthest_proj_vec)
print(furthest_proj_vec)
print()

plt.plot(proj[:,0], proj[:,1], 'bo', label='Projected Data Points')

# replot the closest and furthest points in a different color
plt.plot(closest_proj_vec[:,0], closest_proj_vec[:,1], 'yo', label='Points Closest to Embedding Mean')
plt.plot(furthest_proj_vec[:,0], furthest_proj_vec[:,1], 'ro', label='Points Furthest from Embedding Mean')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVD Projection of BERT Embeddings')
plt.legend()
plt.show()
