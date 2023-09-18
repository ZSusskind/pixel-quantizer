#!/usr/bin/env python3

import argparse
import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatically convert image to pixel art")
    parser.add_argument("input_filename", help="File to convert")
    parser.add_argument("output_filename", help="File to write result to")
    parser.add_argument("--block_size", type=int, default=4, help="Size (nxn) of each pixel in output image")
    parser.add_argument("--palette_size", type=int, default=8, help="Colors in the output image")
    parser.add_argument("--accent_palette_size", type=int, default=0, help="Extra colors to make contrast regions \"pop\"")
    parser.add_argument("--specific_colors", nargs="+", help="Specify extra colors in hex format")
    parser.add_argument("--neighbor_threshold", type=int, default=8, help="Denoising test; the default value seems good")
    args = parser.parse_args()
    return args

def save_file(fname, data, repeat):
    out = np.repeat(np.repeat(data, repeat, axis=0), repeat, axis=1)
    ski.io.imsave(fname, (out*256).astype(np.uint8))

def convert(in_fname, out_fname, block_size, palette_size, accent_size,
            extra_colors, neighbor_threshold):
    image = ski.io.imread(in_fname)
    #target_res = (image.shape[0] // block_size, image.shape[1] // block_size)
    img_x = image.shape[0] // block_size
    img_y = image.shape[1] // block_size
    downscaled = ski.transform.resize(image, (img_x, img_y), anti_aliasing=False)

    linear = downscaled.reshape(-1, 3)

    #forest = IsolationForest(random_state=0, n_estimators=100, contamination=0.02).fit(linear)
    #outlier = forest.predict(linear)

    #luminance = np.array([1, 1, 1])
    #luminance = np.array([0.2126, 0.7152, 0.0722])

    lum_color = linear * luminance
    if palette_size > 0:
        color_kmeans = KMeans(n_clusters=palette_size, random_state=0, n_init="auto").fit(lum_color)
        labels = color_kmeans.predict(lum_color)
        colors = color_kmeans.cluster_centers_[:,0:3]/luminance
        colored = colors[labels]

        badness = ((np.abs(linear-colored)*luminance)**2).sum(axis=1)
        bad_threshold = np.percentile(badness, 98) * .95
        if accent_size > 0:
            accent_linear = linear[badness >= bad_threshold]
            lum_accent = accent_linear * luminance
            accent_kmeans = KMeans(n_clusters=accent_size, random_state=0, n_init="auto").fit(lum_accent)
            labels = accent_kmeans.predict(lum_accent)
            accents = accent_kmeans.cluster_centers_[:,0:3]/luminance
        else:
            accents = np.empty((0,3))
        final_palette = np.concatenate((colors, extra_colors, accents), axis=0)
    else:
        final_palette = extra_colors

    lum_final_palette = final_palette * luminance
    lum_color_deltas = lum_final_palette.reshape(1, -1, 3) - lum_color.reshape(-1, 1, 3)
    lum_delta_norm = (lum_color_deltas**2).sum(axis=2)
    if palette_size > 0:
        lum_delta_norm[(badness<bad_threshold), palette_size+len(extra_colors):] = 1 << 30
    final_colors = lum_delta_norm.argmin(axis=1)
    img_labels = final_colors.reshape(img_x, img_y)
    palettized = final_palette[final_colors].reshape(downscaled.shape)

    #linear_palettized = colors[labels]
    #palettized = linear_palettized.reshape(downscaled.shape)

    #diff = (np.abs(downscaled - palettized) * luminance).sum(axis=2)

    ori_edge = ski.filters.scharr(downscaled).max(axis=2) > 0.1
    pal_edge = ski.filters.scharr(palettized).max(axis=2) > 0.1
    diff_edge = pal_edge & ~ori_edge

    #palettized[diff_edge] = np.array([0.1, 0.9, 0.1])

    # TODO: Eliminate outlying pixels ("outcroppings") first?
    diff = -1
    epoch = 0
    while diff != 0:
        if epoch == 10:
            break
        diff = 0
        for x in range(1, img_x-1):
            for y in range(1, img_y-1):
                if ori_edge[x, y]:
                    continue
                neighbor_labels = [img_labels[i, j] for i in range(x-1,x+2) for j in range(y-1,y+2)]
                count = np.bincount(neighbor_labels)
                if count.max() >= neighbor_threshold:
                    new_label = count.argmax()
                    if new_label != img_labels[x, y]:
                        img_labels[x, y] = new_label
                        palettized[x, y] = final_palette[new_label]
                        diff += 1
        epoch += 1

    for x in range(img_x):
        for y in range(img_y):
            if not diff_edge[x, y]:
                continue
            if (x > 0)\
               and (y%2 == 0)\
               and (img_labels[x-1, y] != img_labels[x, y])\
               and not ori_edge[x-1, y]:
                palettized[x-1, y] = final_palette[img_labels[x, y]]
                palettized[x, y] = final_palette[img_labels[x-1, y]]
            elif (y > 0)\
               and (x%2 == 0)\
               and (img_labels[x, y-1] != img_labels[x, y])\
               and not ori_edge[x, y-1]:
                palettized[x, y-1] = final_palette[img_labels[x, y]]
                palettized[x, y] = final_palette[img_labels[x, y-1]]

    #palettized[outlier.reshape(img_x, img_y)==-1] = np.array([0.1, 0.9, 0.1])
    #palettized[diff > 4*np.median(diff)] = np.array([0.1, 0.9, 0.1])
    #palettized[(badness >= bad_threshold).reshape(img_x, img_y)] = np.array([0.1, 0.9, 0.1])
    #foo = (badness >= bad_threshold).reshape(img_x, img_y)
    #palettized[foo] = downscaled[foo]

    save_file(out_fname, palettized, block_size)

if __name__ == "__main__":
    args = parse_arguments()
    if args.specific_colors is not None:
        extra_colors_list = []
        for c in args.specific_colors:
            assert(6 <= len(c) <= 7)
            if len(c) == 7:
                assert(c[0] == "#")
                c = c[1:]
            cr = int(c[0:2], 16) / 256
            cg = int(c[2:4], 16) / 256
            cb = int(c[4:6], 16) / 256
            extra_colors_list.append([cr, cg, cb])
        extra_colors = np.array(extra_colors_list)
    else:
        extra_colors = np.empty((0,3))
    convert(args.input_filename, args.output_filename, args.block_size, args.palette_size, args.accent_palette_size,
            extra_colors, args.neighbor_threshold)
