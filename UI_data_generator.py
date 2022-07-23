import pygame
import numpy as np
import os
from CNN import *

cwd = os.getcwd()
files = os.listdir(cwd)
print("Files in %r: %s" % (cwd, files))

INP_DIM_X = 28
INP_DIM_Y = 28

W1, b1, W2, b2 = get_trained_weights()
pixels = np.zeros((784, 1))
UNIT = 20
WIDTH, HEIGHT = INP_DIM_X * UNIT, INP_DIM_Y * UNIT
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FONT_SIZE = 30
pygame.font.init()
DISPLAY_FONT = pygame.font.Font("freesansbold.ttf", FONT_SIZE)


def index_on_array(mouse_pos, arr_width):
    x, y = mouse_pos
    col, row = x // UNIT, y // UNIT
    idx = row * arr_width + col
    return idx

def modify_pixels(pixels, arr_width):
    mouse_pos = pygame.mouse.get_pos()
    idx = index_on_array(mouse_pos, arr_width)
    col, row = idx % arr_width, idx // arr_width
    for i in [0, 1, -1, arr_width, -arr_width]:
        if not -1 < idx + i < 784:
            continue
        neigh_col, neigh_row = (idx + i) % arr_width, (idx + i) // arr_width
        d_col = abs(col - neigh_col)
        d_row = abs(row - neigh_row)
        if d_col > 1 or d_row > 1:
            continue
        pixels[idx + i] += .03
        pixels[idx + i] = min(1, pixels[idx + i])

def draw(pixels, prediction):
    WIN.fill((0,0,0))
    for i, pix in enumerate(pixels):
        if not pix:
            continue
        intensity = pix * 255
        col = i % 28
        row = i // 28
        x = col * UNIT
        y = row * UNIT
        pygame.draw.rect(WIN, (intensity, intensity, intensity), (x, y, UNIT, UNIT))
    prediction_surface = DISPLAY_FONT.render(f"My prediction is: {prediction}", False, (130, 130, 130))
    WIN.blit(prediction_surface, (10, 10))
    pygame.display.update()


def main():
    global pixels
    prediction = None
    run = True
    dragging = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    prediction = make_prediction(W1, b1, W2, b2, pixels)
                    prediction = int(prediction)
                    print("PREDICTION: ", prediction)
                if event.key == pygame.K_c:
                    pixels = np.zeros((784, 1))
                    prediction = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
            if event.type == pygame.MOUSEBUTTONUP:
                dragging = False
        if dragging:        
            modify_pixels(pixels, INP_DIM_X)
        draw(pixels, prediction)

main()


