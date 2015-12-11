# -*- coding: utf-8 -*-

# An attempt to find and filter out the outliers from monte carlo image
# Assuming image created by Global Illumination monte carlo simulation

# Load the file
orig_image_file = open('test.ppm', 'r')
oi = orig_image_file
P3 = oi.readline()
dims = map(int, oi.readline().split())
highest_value = int(oi.readline())

r_lines = []
g_lines = []
b_lines = []

# Extract color values
for curr_line in oi:
    values = map(int, curr_line.split())
    r_lines += [values[::3]]
    g_lines += [values[1::3]]
    b_lines += [values[2::3]]
   
# Save into new files
def write_pgm(name,w,h,buffer):
    wf = open(name + '.pgm', 'w')
    wf.write('P2\n')
    wf.write("%d %d\n" % (w, h))
    wf.write("%d\n" % (highest_value))
    for color_line in buffer:
        wf.write(' '.join(map(str, color_line)) + "\n")
    wf.close()

# View each channel (R G B) independently
write_pgm('red',dims[0], dims[1], r_lines)
write_pgm('green',dims[0], dims[1], g_lines)
write_pgm('blue',dims[0], dims[1], b_lines)

# Merge back together to verify we did everything right
import itertools
def write_ppm(name,w,h,b1,b2,b3):
    wf = open(name + '.ppm', 'w')
    wf.write('P3\n')
    wf.write("%d %d\n" % (w, h))
    wf.write("%d\n" % (highest_value))
    for rline,gline,bline in itertools.izip(b1,b2,b3):
        for r,g,b in itertools.izip(rline,gline,bline):
            wf.write('%d %d %d ' % (r,g,b))
        wf.write("\n")
    wf.close()

write_ppm('merged',dims[0],dims[1],r_lines,g_lines,b_lines)

import colorsys

# Convert to HSV
hsv_rows = []
range01 = lambda v: v * (1.0 / 255.0)
for rline,gline,bline in itertools.izip(r_lines,g_lines,b_lines):
    hsv_row = []
    for r,g,b in itertools.izip(map(range01,rline),
                                map(range01,gline),
                                map(range01,bline)):
        hsv_row += colorsys.rgb_to_hsv(r,g,b)
    hsv_rows += [hsv_row]

# Save into separate H S V files
h_rows = []
s_rows = []
v_rows = []
for hsv_row in hsv_rows:
    values = hsv_row
    h_rows += [values[::3]]
    s_rows += [values[1::3]]
    v_rows += [values[2::3]]

import numpy as np
range255int = lambda rows: (np.matrix(rows) * 255.0001).astype(int).tolist()
write_pgm('hue',dims[0], dims[1], range255int(h_rows))
write_pgm('sat',dims[0], dims[1], range255int(s_rows))
write_pgm('val',dims[0], dims[1], range255int(v_rows))
# View those files - is the problem easy to see?

# Make a map which clearly highlights the problem areas (the fire flies)
find_flies = [[-0.125, -0.125, -0.125],[-0.125, 1.0, -0.125],[-0.125, -0.125, -0.125]]
find_flies = np.matrix(find_flies)
vmat = np.matrix(v_rows)
vsec = vmat[180:200, 180:200]

import scipy.ndimage
flies_sec = scipy.ndimage.filters.correlate(vsec, find_flies)
flies_mat = scipy.ndimage.filters.correlate(vmat, find_flies)

write_pgm('vsec',20, 20, range255int(vsec))

range255int0cap = lambda rows: (np.matrix(rows).clip(min=0) * 255.0001).astype(int).tolist()

write_pgm('flies_sec',20, 20, range255int0cap(flies_sec))

write_pgm('flies',dims[0], dims[1], range255int0cap(flies_mat))

flies_mat2 = scipy.ndimage.filters.correlate(flies_mat, find_flies)
write_pgm('flies2',dims[0], dims[1], range255int0cap(flies_mat2))

find_flies2 = (find_flies * 2.0)
find_flies2[find_flies2 > 0] = 1
flies_mat2a = scipy.ndimage.filters.correlate(flies_mat, find_flies2)
write_pgm('flies2a',dims[0], dims[1], range255int0cap(flies_mat2a))


only_neighbors_filter = np.matrix([[0.125, 0.125, 0.125],[0.125, 0.0, 0.125],[0.125, 0.125, 0.125]])
neighbors_mat = scipy.ndimage.filters.correlate(vmat, only_neighbors_filter)
write_pgm('neighbors',dims[0], dims[1], range255int0cap(neighbors_mat))

diff_mat = vmat - neighbors_mat
write_pgm('diff',dims[0], dims[1], range255int0cap(diff_mat))

#tm2 = (flies_sec * 100.0).astype(int)
#tm3 = tm2[0:6,0:6]
#printit = lambda x: x[4]
#scipy.ndimage.filters.generic_filter(tm3, printit, size=3)
#scipy.ndimage.filters.generic_filter(tm3, printit, footprint=[[1,1],[1,1]])

# -- Look for numpy matrix stuff to run kernels on our data --

def min_diff_abs(values):
    pixel_value = values[len(values)/2]
    neighbors = np.delete(values, len(values)/2)
    diffs = map(lambda x: pixel_value - x, neighbors)
    return min(abs(np.array(diffs)))

def min_diff(values):
    pixel_value = values[len(values)/2]
    neighbors = np.delete(values, len(values)/2)
    diffs = map(lambda x: pixel_value - x, neighbors)
    return min(np.array(diffs))

#scipy.ndimage.filters.generic_filter(tm3, min_diff, size=3)


min_diff_mat = scipy.ndimage.filters.generic_filter(vmat, min_diff, size=3)
write_pgm('min_diff2',dims[0], dims[1], range255int0cap(min_diff_mat))


# Use that map to fix our image
# Fixing will probably be just blowing away our pixel and merging neighbor
# information.
from compiler.ast import flatten
def blur_out_fireflies(img,ffmap,thresh):
    new_img = []
    new_img += [img[0]]
    for i in range(1,1+len(img[1:-1])):
        new_row = []
        new_row += [img[i][0]]
        for j in range(1,1+len(img[i][1:-1])):
            if (ffmap[i][j] <= thresh):
                new_row += [img[i][j]]
            else:
                new_row += [np.average(flatten([img[i-1][j-1:j+1],img[i+1][j-1:j+1],img[i][j-1],img[i][j+1]]))]
        new_row += [img[i][-1]]
        new_img += [new_row]
    new_img += [img[-1]]
    return new_img

fixed_v = blur_out_fireflies(vmat.tolist(),min_diff_mat.tolist(),0.25)
write_pgm('fixed_flies',dims[0], dims[1], range255int0cap(fixed_v))

# Recombine into RGB
rgb_rows = []
for h_row,s_row,v_row in itertools.izip(h_rows,s_rows,fixed_v):
    rgb_row = []
    for h,s,v in itertools.izip(h_row, s_row, v_row):
        rgb_row += colorsys.hsv_to_rgb(h,s,v)
    rgb_rows += [rgb_row]

rgb255_rows = range255int(rgb_rows)

r_rows = []
g_rows = []
b_rows = []
for rgb_row in rgb255_rows:
    values = rgb_row
    r_rows += [values[::3]]
    g_rows += [values[1::3]]
    b_rows += [values[2::3]]

# Save out the new image - DONE
write_ppm('fixed_image',dims[0],dims[1],r_rows,g_rows,b_rows)

