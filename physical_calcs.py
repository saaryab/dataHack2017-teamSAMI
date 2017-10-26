"""
As soon the Z acceleration is -10m/s, 
v = a*t 
delta on every two points, divide by delta of the times, and that's the acceleration
"""
import math
from utils import load_dataset
def _calc_acceleration(v1, v2, t1, t2):
    return (v2 - v1) / (t2-t1)

TIME_IDX, POSX_IDX, POSY_IDX, POSZ_IDX, VELX_IDX, VELY_IDX, VELZ_IDX = range(7)

def calc_acceleration(vector1, vector2):
    x_acc = _calc_acceleration(vector1[VELX_IDX],
                      vector2[VELX_IDX],
                      vector1[TIME_IDX],
                      vector2[TIME_IDX])
    y_acc = _calc_acceleration(vector1[VELY_IDX],
                      vector2[VELY_IDX],
                      vector1[TIME_IDX],
                      vector2[TIME_IDX])
    z_acc = _calc_acceleration(vector1[VELZ_IDX],
                      vector2[VELZ_IDX],
                      vector1[TIME_IDX],
                      vector2[TIME_IDX])
    return (x_acc, y_acc, z_acc)

"""return a list of tuples, where each tuple is an acceleration of a rocket (x,y,z)"""
def get_all_accelerations(rocket_samples):
    accelerations = []
    for sample_idx in list(range(len(rocket_samples) - 1)):
        acc = calc_acceleration(rocket_samples[sample_idx], rocket_samples[sample_idx+1])
        accelerations.append(acc)
    return accelerations


"""Return 3 lists of all the acclerations of a rocket."""
def get_all_accelerations_by_vector_direction(rocket_samples):

    x_accs = []
    y_accs = []
    z_accs = []
    for sample_idx in list(range(len(rocket_samples) - 1)):
        x_acc, y_acc, z_acc = calc_acceleration(rocket_samples[sample_idx], rocket_samples[sample_idx+1])
        x_accs.append(x_acc)
        y_accs.append(y_acc)
        z_accs.append(z_acc)
    return x_accs, y_accs, z_accs



# dataset_1 = load_dataset(index=1)
