"""'''
Created Date: May 8, 2020
Author: Nana Chang
Description:
This piece of code aimed to model the received signal to each microphone for ReSpeaker with relative location between
the sources an the microphone array known.

- Input & Parameters:
    .  input_file_name: german_speech_8000.wav
    .  source location: r for distance and direction in degree
- Output:
    .  german_speech_8000_delay_1s_ma.wav


"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra


def ideal_process(fs, input, r, direction, output_file_name=None, noise=True, sigma=500, writeFile=True):
    source_location = np.array([r * np.cos(np.pi * direction / 180.), r * np.sin(np.pi * direction / 180.)])
    path = os.path.dirname(__file__)
    micArray = np.array([[0.029149, 0.029149],
                         [-0.029149, 0.029149],
                         [-0.029149, -0.029149],
                         [0.029149, -0.029149]])
    c = 340.0 # sound speed

    delay = np.zeros(4)
    added_frame = np.zeros(4, dtype = 'int16')
    for i, mic in enumerate(micArray):
        a = source_location - mic
        delay[i] = np.linalg.norm(a) / c
        added_frame[i] = int(fs * delay[i])

    output = np.zeros((4, len(input) + np.max(added_frame)), dtype=input.dtype)

    for i in range(4):
        output[i, added_frame[i]:added_frame[i] + len(input)] = input

    output = output.T

    if noise:
        added_noise = np.random.normal(0, sigma, output.shape)
        output = output + added_noise

    output = output.astype(np.int16)

    if writeFile:
        wavfile.write(path + '/input/' + output_file_name, fs, output)

    return output


def pyroom_simulation_process(fs, input, r, direction, output_file_name=None, noise=True, sigma=500, writeFile=True, dimension=2):
    path = os.path.dirname(__file__)

    floor = np.array([[2*r, -2*r, -2*r, 2*r], [2*r, 2*r, -2*r, -2*r]])
    micArray = np.array([[0.029149, 0.029149],
                         [-0.029149, 0.029149],
                         [-0.029149, -0.029149],
                         [0.029149, -0.029149]])

    bf = pra.Beamformer(micArray.T, fs)
    room = pra.Room.from_corners(floor, 1., fs, mics=bf)
    if dimension == 3:
        room.extrude(1.5)
    source_location = np.array([r * np.cos(np.pi * direction / 180.), r * np.sin(np.pi * direction / 180.)])
    room.add_source(source_location, signal=input)
    if noise:
        room.simulate(snr=20)
    else:
        room.simulate()
    output = bf.signals.T * np.max(input) / np.max(bf.signals)
    output = output.astype(np.int16)

    if writeFile:
        wavfile.write(path + '/input/' + output_file_name, fs, output)
    return output


def plot(input, output):
    plt.subplot(211)
    plt.plot(input)
    plt.title('input data')
    plt.subplot(212)
    plt.plot(output)
    plt.title('output data')
    plt.show()


if __name__ == "__main__":
    input_file_name = 'german_speech_8000_withDelay.wav'
    r = 10.0
    direction = 135.0

    path = os.path.dirname(__file__)
    fs, input = wavfile.read(path + '/input/' + input_file_name)

    output = ideal_process(fs, input, r, direction, 'DoA_manuallyGeneratedFile.wav', noise=True, writeFile=True)
    output2 = pyroom_simulation_process(fs, input, r, direction, 'DoA_manuallyGeneratedFile.wav', noise=True, writeFile=True)

    plot(input, output)
    plot(input, output2)
