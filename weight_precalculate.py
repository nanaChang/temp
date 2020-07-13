"""
Author: Nana Chang
Date: 2020/07/10
Description: This piece of code aimed to calculate the weight of MVDR beamformer from each direction and output as a npy
file for further BF implementation so as to run BF algorithm in realtime.

"""
import pyroomacoustics as pra
import numpy as np

# x = np.zeros((2, 4))
# x[0, 1] = 5
# x[1, 2] = 4
# x[1, 3] = 15
# print(x)

# np.save('test.npy', x)

# a = np.load('test.npy')
# print('---------------------')
# print(a)


def weight_calculate(alg_type):
    R = np.array([[-0.029149, -0.029149],
                  [-0.029149, 0.029149],
                  [0.029149, 0.029149],
                  [0.029149, -0.029149]]).T

    floor = np.array([[-5, 5, 5, -5], [5, 5, -5, -5]])
    Fs = 16000

    bf = pra.Beamformer(R, Fs)
    sigma2_n = 5e-7
    delay = bf.Lg / 2 / bf.fs
    n = 72
    filter = np.zeros((n, 4, bf.Lg), dtype=np.complex_)
    weight = np.zeros((n, 4, bf.N // 2 + 1), dtype=np.complex_)
    if alg_type is 'direct_mvdr':
        for i in range(n):
            print(i)
            direc = i * 2 / n * np.pi
            room = pra.Room.from_corners(floor, fs=Fs, max_order=1, absorption=0.1, mics=bf)
            room.add_source([np.cos(direc), np.sin(direc)])
            room.add_source([np.cos(direc + np.pi), np.sin(direc + np.pi)])

            bf.rake_mvdr_filters(room.sources[0],
                                 room.sources[1],
                                 R_n=sigma2_n * np.eye(bf.Lg * bf.M),
                                 delay=delay)

            bf.weights_from_filters()
            filter[i, :, :] = bf.filters
            weight[i, :, :] = bf.weights

            del room

    elif alg_type is 'MVDR_set_interference':
        for i in range(n):
            print(i)
            direc = i * 2 / n * np.pi
            room = pra.Room.from_corners(floor, fs=Fs, max_order=1, absorption=0.1, mics=bf)
            room.add_source([np.cos(direc), np.sin(direc)])
            room.add_source([np.cos(direc + np.pi), np.sin(direc + np.pi)])

            room.image_source_model()

            x = np.linspace(0, 4, 5) * np.pi / 4
            room.sources[1].images = np.array([np.cos(direc + np.pi / 2 + x), np.sin(direc + np.pi / 2 + x)]) * 5
            room.sources[1].damping = np.ones(5)

            bf.rake_mvdr_filters(room.sources[0][0:1],
                                 room.sources[1],
                                 R_n=sigma2_n * np.eye(bf.Lg * bf.M),
                                 delay=delay)
            bf.weights_from_filters()
            filter[i, :, :] = bf.filters
            weight[i, :, :] = bf.weights

            del room
    elif alg_type is 'DAS':
        for i in range(n):
            print(i)
            direc = i * 2 / n * np.pi
            room = pra.Room.from_corners(floor, fs=Fs, max_order=1, absorption=0.1, mics=bf)
            room.add_source([np.cos(direc), np.sin(direc)])
            room.add_source([np.cos(direc + np.pi), np.sin(direc + np.pi)])
            room.image_source_model()

            x = np.linspace(0, 4, 5) * np.pi / 4
            room.sources[1].images = np.array([np.cos(direc + np.pi / 2 + x), np.sin(direc + np.pi / 2 + x)]) * 5
            room.sources[1].damping = np.ones(5)

            bf.rake_delay_and_sum_weights(room.sources[0][0:1],
                                          room.sources[1],
                                          R_n=sigma2_n * np.eye(bf.Lg * bf.M))
            bf.filters_from_weights()

    np.save(alg_type + '_filter.npy', filter)
    np.save(alg_type + '_weight.npy', weight)



if __name__ == '__main__':
    alglist = ['direct_mvdr',
               'MVDR_set_interference',
               'DAS']
    for alg_type in alglist:
        print(alg_type)
        weight_calculate(alg_type)
