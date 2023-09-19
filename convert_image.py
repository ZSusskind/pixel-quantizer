#!/usr/bin/env python3

import argparse
import numpy as np
import skimage as ski
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from numba import njit, prange

import vcolor

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automatically convert image to pixel art")
    parser.add_argument("input_filename", help="File to convert")
    parser.add_argument("output_filename", help="File to write result to")
    parser.add_argument("--block_size", type=int, default=4,
                        help="Edge size of output image pixel in original image pixels")
    parser.add_argument("--palette_size", type=int, default=8,
                        help="Total unique colors to use in the output image")
    parser.add_argument("--accent_size", type=int, default=0,
                        help="Reserves palette colors for pixels which can't be represented well "\
                        "by the first-pass palette generation; good for capturing fine details")
    parser.add_argument("--specific_colors", nargs="+",
                        help="Specify palette colors in hex format")
    parser.add_argument("--specific_accents", nargs="+",
                        help="Ditto, but for accents")
    parser.add_argument("--denoise", type=int, default=0,
                        help="Attempts to smooth spurious \"outcroppings\" on palette color region boundaries; "\
                        "higher values will try more and larger regions")
    parser.add_argument("--dither", type=int, default=0, choices=[0, 1, 2, 4],
                        help="Use dithering to smooth palette color transitions; higher levels give stronger effects")
    parser.add_argument("--quantize", type=int, nargs=3, default=[8, 8, 8],
                        help="Specify bit depth separately for R/G/B channels (1-8); "\
                        "with very low values, the final palette may have fewer colors than desired")
    parser.add_argument("--accent_percentile", type=float, default=98.0,
                        help="Target \"badness\" percentile score for accent features. Setting this lower will "
                        "use accent colors for a larger fraction of pixels. Range: (0,100)")
    parser.add_argument("--resample", action="store_true",
                        help="Resample color space after removing accent pixels")
    parser.add_argument("--color_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Skew the relative importance of the RGB channels")
    parser.add_argument("--saturate", type=float, default=1.0,
                        help="Change saturation of image by factor; saturating an image can improve quality")
    parser.add_argument("--tint", type=float, nargs=2, default=[0.0, 0.0],
                        help="Shift towards <hue [0, 360)> with <intensity [0, 1]>")

    args = parser.parse_args()
    return args

def save_file(fname, data, repeat):
    out = np.repeat(np.repeat(data, repeat, axis=0), repeat, axis=1)
    ski.io.imsave(fname, (out*256).astype(np.uint8))

def saturate_and_tint(linear, saturation, tint_hue, tint_strength):
    assert(saturation >= 0.0)
    assert(0 <= tint_hue < 360)
    assert(0 <= tint_strength <= 1)
    hls = vcolor.vector_rgb_to_hls(linear)
    if saturation <= 1.0:
        hls[2] = saturation*hls[2]
    else:
        hls[2] = hls[2]**(1/saturation)
    hls[0] = (1.0 - tint_strength) * hls[0] + tint_strength * (tint_hue / 360)
    return vcolor.vector_hls_to_rgb(hls)

def make_palette(num_colors, data, extra_colors):
    if num_colors > 0:
        kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=0, n_init="auto", batch_size=1<<17)
        kmeans = kmeans.fit(data)
        palette = kmeans.cluster_centers_
        if specific_colors is not None:
            palette = np.concatenate((palette, extra_colors), axis=0)
    else:
        assert((extra_colors is not None) and (len(extra_colors) > 0))
        palette = extra_colors
    return palette

def quantize_to_palette(palette, colors, mask=None, mask_start=None):
    deltas = palette.reshape(1, -1, 3) - colors.reshape(-1, 1, 3)
    delta_norm = (deltas**2).sum(axis=2)
    if mask is not None:
        delta_norm[mask, mask_start:] = np.inf
    labels = delta_norm.argmin(axis=1)
    return labels

@njit(parallel=True)
def run_denoising(img_x, img_y, ori_edge, img_labels, kernel_size, threshold):
    assert(kernel_size%2 == 1)
    kernel_pad = kernel_size // 2
    done = False
    iter = 0
    while not done:
        if iter == 20:
            break
        done = True
        for x in prange(kernel_pad, img_x-kernel_pad):
            for y in range(kernel_pad, img_y-kernel_pad):
                if ori_edge[x, y]:
                    continue
                #neighbor_edges = ori_edge[x-kernel_pad:x+kernel_pad+1,
                #                          y-kernel_pad:y+kernel_pad+1].flatten()
                #if neighbor_edges.any():
                #    continue
                neighbor_labels = img_labels[x-kernel_pad:x+kernel_pad+1,
                                             y-kernel_pad:y+kernel_pad+1].flatten()
                #values, counts = np.unique(neighbor_labels, return_counts=True)
                values = np.unique(neighbor_labels)
                max_value = 0
                max_count = 0
                for v in values:
                    count = (neighbor_labels == v).sum()
                    if count > max_count:
                        max_count = count
                        max_value = v

                #max_count = counts.max()
                #max_value = values[counts.argmax()]
                if max_count >= threshold:
                    new_label = max_value
                    if new_label != img_labels[x, y]:
                        img_labels[x, y] = new_label
                        done = False
        iter += 1

def convert(in_fname, out_fname, block_size, palette_size, accent_size,
            specific_colors, specific_accents, denoise, dither, quantize,
            accent_percentile, resample, color_weights,
            saturation, tint_hue, tint_strength):
    quantize = np.array(quantize)
    color_weights = np.array(color_weights)

    assert(accent_size <= palette_size)
    assert(quantize.min() >= 1)
    assert(quantize.max() <= 8)
    assert(0 < accent_percentile < 100)
    if (denoise == 0) and (dither > 1):
        print("NOTE: Using dithering > 1 without denoising is not recommended")

    image = ski.io.imread(in_fname)

    if not all((image.shape[i] / block_size).is_integer() for i in range(2)):
        print("Image size is not an integer multiple of block size; output image resolution will not match")
    img_x = image.shape[0] // block_size
    img_y = image.shape[1] // block_size
    downscaled = ski.transform.resize(image, (img_x, img_y), anti_aliasing=False)

    linear = downscaled.reshape(-1, 3)
    if (saturation != 1.0) or (tint_strength != 0.0):
        print("Adjust saturation and tint")
        linear = saturate_and_tint(linear, saturation, tint_hue, tint_strength)

    weighted = linear * color_weights
    
    print("Determining base palette")
    base_palette_size = palette_size - accent_size
    base_palette = make_palette(base_palette_size, weighted, specific_colors)

    if (accent_size > 0) or ((specific_accents is not None) and (len(specific_accents) > 0)):
        print("Evaluating base palette badness")
        base_labels = quantize_to_palette(base_palette, weighted)
        base_palettized_linear = base_palette[base_labels]
        badness = ((base_palettized_linear-weighted)**2).sum(axis=1)
        bad_threshold = np.percentile(badness, accent_percentile) * .95
        bad_mask = badness >= bad_threshold
        print(f"  Mean badness of {round(badness.mean(), 3)}; "\
              f"{round(bad_mask.mean(), 3)} of pixels exceed threshold {round(bad_threshold, 3)}")

        print("Determining accent palette")
        accent_palette = make_palette(accent_size, weighted[bad_mask], specific_accents)
        if resample:
            base_palette = make_palette(base_palette_size, weighted[~bad_mask], specific_colors) # rerun w/o bad pixels
    else:
        bad_mask = np.zeros((len(weighted),)).astype(bool)
        accent_palette = np.empty((0, 3))

    weighted_palette = np.concatenate((base_palette, accent_palette) , axis=0)
    print("Applying palette to image")
    labels = quantize_to_palette(weighted_palette, weighted, ~bad_mask, len(base_palette))
    img_labels = labels.reshape(img_x, img_y)
    final_palette = weighted_palette / color_weights
    final_palette = np.clip(np.round(final_palette * ((1<<quantize)-1)) / ((1<<quantize)-1), 0.0, 0.9999)
    assert(final_palette.min() >= 0)
    assert(final_palette.max() < 1)
    palettized = final_palette[labels].reshape(downscaled.shape)

    if (denoise > 0) or (dither > 0):
        ori_edge = ski.filters.scharr(downscaled).max(axis=2) > 0.1

        if dither >= 2:
            print("Running 2:2 dithering")
            assignment_result = weighted_palette[img_labels.flatten()]
            assignment_error = assignment_result - weighted
            assignment_badness = (assignment_error**2).sum(axis=1)
            
            assignment_correction = weighted - assignment_error
            #corrected_labels = quantize_to_palette(weighted_palette, assignment_correction,
            #                                       ~bad_mask, len(base_palette))
            corrected_labels = quantize_to_palette(base_palette, assignment_correction)
            corrected_img_labels = corrected_labels.reshape(img_x, img_y)
            corrected_error = (weighted_palette[corrected_labels] - weighted)

            half_corrected_error = (corrected_error + assignment_error) / 2
            half_corrected_badness = (half_corrected_error**2).sum(axis=1)
            half_correction_mask = (half_corrected_badness < assignment_badness).reshape(img_x, img_y)
            #half_correction_mask = np.logical_and(half_correction_mask, ~ori_edge)

            if dither >= 4:
                print("Running 3:1 dithering")
                quarter_corrected_error = (corrected_error + 3*assignment_error) / 4
                quarter_corrected_badness = (quarter_corrected_error**2).sum(axis=1)
                best_badness = np.minimum(assignment_badness, half_corrected_badness)
                quarter_correction_mask = (quarter_corrected_badness < best_badness).reshape(img_x, img_y)
                #quarter_correction_mask = np.logical_and(quarter_correction_mask, ~ori_edge)
                half_correction_mask[quarter_correction_mask] = False
            else:
                quarter_correction_mask = np.zeros((img_x, img_y)).astype(bool)

            assert(not np.logical_and(half_correction_mask, quarter_correction_mask).any())
            
            # Create "meta" labels for dithered colors
            img_labels[half_correction_mask] += ((corrected_img_labels+1)\
                * len(final_palette))[half_correction_mask]
            img_labels[quarter_correction_mask] += ((corrected_img_labels+len(final_palette)+1)\
                * len(final_palette))[quarter_correction_mask]

        if (denoise > 0):
            print("Running denoising")
            for i in range(denoise):
                k = (2*i) + 3
                t = int(np.round(.85*k**2))
                #print(f"Running denoising with kernel {k} and threshold {t}")
                run_denoising(img_x, img_y, ori_edge, img_labels, k, t)
        
        if dither == 1:
            print("Running edge dithering")
            pal_edge = ski.filters.scharr(palettized).max(axis=2) > 0.1
            diff_edge = pal_edge & ~ori_edge # Spurious edges added by palettization

            new_img_labels = img_labels.copy()
            for x in range(img_x):
                for y in range(img_y):
                    if not diff_edge[x, y]:
                        continue
                    if (x > 0)\
                       and (y%2 == 0)\
                       and (img_labels[x-1, y] != img_labels[x, y])\
                       and not ori_edge[x-1, y]:
                        new_img_labels[x-1, y] = img_labels[x, y]
                        new_img_labels[x, y] = img_labels[x-1, y]
                    elif (y > 0)\
                       and (x%2 == 0)\
                       and (img_labels[x, y-1] != img_labels[x, y])\
                       and not ori_edge[x, y-1]:
                        new_img_labels[x, y-1] = img_labels[x, y]
                        new_img_labels[x, y] = img_labels[x, y-1]
            img_labels = new_img_labels
        
        half_correction_mask = np.logical_and((img_labels >= len(final_palette)),
                                              (img_labels < len(final_palette)*(len(final_palette)+1)))
        quarter_correction_mask = img_labels >= len(final_palette)*(len(final_palette)+1)
        correction_mask = np.logical_or(half_correction_mask, quarter_correction_mask)
    
        corrected_img_labels = ((img_labels // len(final_palette))-1) % len(final_palette)
        img_labels %= len(final_palette)

        half_mask_x = (-1) ** np.arange(img_x)
        half_mask_y = (-1) ** np.arange(img_y)
        half_diag_mask = (1 - (half_mask_x.reshape(-1, 1) * half_mask_y.reshape(1, -1))).astype(bool)
        half_sort_mask = np.where(img_labels < corrected_img_labels, half_diag_mask, ~half_diag_mask)
        half_dither_mask = np.logical_and(half_correction_mask, half_sort_mask)
                
        quarter_mask_x = (1 - ((-1) ** np.arange(img_x)))//2
        quarter_mask_y = (1 - ((-1) ** np.arange(img_y)))//2
        quarter_diag_mask = (quarter_mask_x.reshape(-1, 1) * (1-quarter_mask_y).reshape(1, -1)).astype(bool)
        inv_quarter_diag_mask = (quarter_mask_x.reshape(-1, 1) * quarter_mask_y.reshape(1, -1)).astype(bool)
        quarter_sort_mask = np.where(img_labels < corrected_img_labels, quarter_diag_mask, inv_quarter_diag_mask)
        assert((np.logical_and(quarter_sort_mask, half_sort_mask) == quarter_sort_mask).all())
        quarter_dither_mask = np.logical_and(quarter_correction_mask, quarter_sort_mask)

        dither_mask = np.logical_or(half_dither_mask, quarter_dither_mask)

        img_labels = np.where(dither_mask, corrected_img_labels, img_labels)
        palettized = final_palette[img_labels.flatten()].reshape(downscaled.shape)

    print("All done!")
    save_file(out_fname, palettized, block_size)

def parse_colors(colors):
    color_list = []
    for c in colors:
        assert(6 <= len(c) <= 7)
        if len(c) == 7:
            assert(c[0] == "#")
            c = c[1:]
        cr = int(c[0:2], 16) / 256
        cg = int(c[2:4], 16) / 256
        cb = int(c[4:6], 16) / 256
        color_list.append([cr, cg, cb])
    return np.array(color_list)

if __name__ == "__main__":
    args = parse_arguments()
    specific_colors = parse_colors(args.specific_colors) if args.specific_colors is not None else None
    specific_accents = parse_colors(args.specific_accents) if args.specific_accents is not None else None
    convert(args.input_filename, args.output_filename, args.block_size, args.palette_size, args.accent_size,
            specific_colors, specific_accents, args.denoise, args.dither, args.quantize,
            args.accent_percentile, args.resample, args.color_weights,
            args.saturate, *args.tint)
