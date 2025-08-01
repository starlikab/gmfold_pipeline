from __future__ import print_function
from __future__ import division
from builtins import zip
from past.builtins import basestring

import Bio
from forgi.graph.bulge_graph import BulgeGraph
import forgi.graph.residue as fgr
import forgi.threedee.model.coarse_grain as ftmc
import forgi.threedee.utilities.pdb as ftup
import forgi.threedee.utilities.vector as ftuv
import math
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
import colorsys
from matplotlib import cm

import pandas as pd
import random
import re



log = logging.getLogger(__name__)


def remove_spaces_and_convert_to_upper(input_string, remove_first = 5):
    #input_string = input_string.upper()
    input_string = input_string.replace(" ", "")
    input_string = input_string.replace("\u200b", "")
    return input_string[remove_first:]

def remove_spaces(input_string):
    #input_string = input_string.upper()
    input_string = input_string.replace(" ", "")
    input_string = input_string.replace("\u200b", "")
    return input_string


def change_name( input_sequence, seed ):
    
    return input_sequence + f'_seed_{seed}'



def check_char(sequence):
    mapping = {'A': 0, 'a': 0, 'T': 3, 't': 3, 'C': 1, 'c':1, 'G': 2, 'g':2}
    flag = True
    for char in sequence:
        if char not in mapping:
             flag = False

    return flag

def check_if_string(sequence):
    if isinstance(sequence, str):
        return True
    else:
        return False


def check_structure(input_string):
    pattern = r'^.*?____(.*?)____.*?$'
    if re.match(pattern, input_string):
        return True
    else:
        return False
    
def check_empty_or_short(input_string, threshold = 20): 
    if len(input_string) >=threshold:
        return True
    else:
        return False
    
def find_substrings(input_string):
    return re.findall(r'____(.*?)____', input_string)[0]

def attach_stems(main_string, stem1='GGGACGAC', stem2 ='GTCGTCCC' ):
    return  stem1 + main_string + stem2

# New Function for Coloring Multi-Strand Structures
# Modified Function "plot_rna" from https://viennarna.github.io/forgi/index.html

def get_strand_colors(n, cmap_name='tab10'):
    """Return a list of distinct RGB colors for `n` strands."""
    cmap = cm.get_cmap(cmap_name, n)
    return [cmap(i)[:3] for i in range(n)]


def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None,labels=[], **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    :param x,y: scalar or array_like, shape (n, )
                Input data
    :param s: scalar or array_like, shape (n, )
              Radius of circle in data scale (ie. in data unit)
    :param c: color or sequence of color, optional, default : 'b'
              `c` can be a single color format string, or a sequence of color
              specifications of length `N`, or a sequence of `N` numbers to be
              mapped to colors using the `cmap` and `norm` specified via kwargs.
              Note that `c` should not be a single numeric RGB or
              RGBA sequence because that is indistinguishable from an array of
              values to be colormapped.  `c` can be a 2-D array in which the
              rows are RGB or RGBA, however.
    :param ax: Axes object, optional, default: None
               Parent axes of the plot. It uses gca() if not specified.
    :param vmin, vmax: scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.  (Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.)
    :param kwargs: `~matplotlib.collections.Collection` properties
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap

    :returns: paths : `~matplotlib.collections.PathCollection`


    Examples ::

        a = np.arange(11)
        circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')

    This code by ??? is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)

    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    #import matplotlib.colors as colors

    if ax is None:
        ax = plt.gca()

    if isinstance(c, basestring):
        color = c     # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)

    if np.isscalar(x):
        patches = [Circle((x, y), s), ]
    elif np.isscalar(s):
        patches = [Circle((x_, y_), s) for x_, y_ in zip(x, y)]
    else:
        patches = [Circle((x_, y_), s_) for x_, y_, s_ in zip(x, y, s)]

    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _clashfree_annot_pos(pos, coords):
    for c in coords:
        dist = ftuv.vec_distance(c, pos)
        #log.debug("vec_dist=%s", dist)
        if dist<14:
            return False
    return True

def _annotate_rna_plot(ax, cg, coords, annotations, text_kwargs):
    # Plot annotations
    annot_dict = { elem:elem for elem in cg.defines}
    if annotations is None:
        annot_dict={elem:"" for elem in cg.defines}
    else:
        annot_dict.update(annotations)
    stem_coords = {}
    for stem in cg.stem_iterator():
        stem_start =  np.mean([coords[cg.defines[stem][0]-1],
                               coords[cg.defines[stem][3]-1]],
                               axis=0)
        stem_end =  np.mean([coords[cg.defines[stem][1]-1],
                               coords[cg.defines[stem][2]-1]],
                               axis=0)
        stem_center = np.mean([stem_start, stem_end], axis=0)
        stem_coords[stem]=(stem_start, stem_center, stem_end)
        if annot_dict[stem]:
            stem_vec = stem_end - stem_start
            norm_vec = (stem_vec[1], -stem_vec[0])
            norm_vec/=ftuv.magnitude(norm_vec)
            annot_pos = np.array(stem_center)+23*norm_vec
            #log.debug("Checking clashfree for %s, %s", stem, annot_pos)
            if not _clashfree_annot_pos(annot_pos, coords):
                log.debug("Cannot annotate %s as %s ON THE RIGHT HAND SIDE, because of insufficient space. Trying left side...", stem, annot_dict[stem])
                annot_pos = np.array(stem_center)-23*norm_vec
                #log.debug("Checking clashfree OTHER SIDE for %s, %s", stem, annot_pos)
                if not _clashfree_annot_pos(annot_pos, coords):
                    log.info("Cannot annotate %s as '%s', because of insufficient space.", stem, annot_dict[stem])
                    annot_pos = None
            #log.debug("%s", annot_pos)
            if annot_pos is not None:
                ax.annotate(annot_dict[stem], xy=annot_pos,
                            ha="center", va="center", **text_kwargs )
    for hloop in cg.hloop_iterator():
        hc = []
        for nt in cg.define_residue_num_iterator(hloop, adjacent=True):
            hc.append(coords[nt-1])
        annot_pos = np.mean(hc, axis=0)
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[hloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.info("Cannot annotate %s as '%s' ON THE INSIDE, because of insufficient space. Trying outside...", hloop, annot_dict[hloop])
            nt1, nt2 = cg.define_a(hloop)
            start = np.mean([coords[nt1-1], coords[nt2-1]], axis=0)
            vec = annot_pos-start
            annot_pos = annot_pos+vec*3
            if _clashfree_annot_pos(annot_pos, coords):
                ax.annotate(annot_dict[hloop], xy=annot_pos,
                            ha="center", va="center", **text_kwargs )
            else:
                log.info("Cannot annotate %s as '%s', because of insufficient space.", hloop, annot_dict[hloop])
    for iloop in cg.iloop_iterator():
        s1, s2 = cg.connections(iloop)
        annot_pos = np.mean([ stem_coords[s1][2], stem_coords[s2][0]], axis=0)
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[iloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.debug("Cannot annotate %s as '%s' ON THE INSIDE, because of insufficient space. Trying outside...", iloop, annot_dict[iloop])
            loop_vec = stem_coords[s2][0] - stem_coords[s1][2]
            norm_vec = (loop_vec[1], -loop_vec[0])
            norm_vec/=ftuv.magnitude(norm_vec)
            annot_pos_p = np.array(annot_pos)+25*norm_vec
            annot_pos_m = np.array(annot_pos)-25*norm_vec
            # iloops can be asymmetric (more nts on one strand.)
            # plot the label on the strand with more nts.
            plus=0
            minus=0
            for nt in cg.define_residue_num_iterator(iloop):
                if ftuv.vec_distance(annot_pos_p, coords[nt-1])<ftuv.vec_distance(annot_pos_m, coords[nt-1]):
                    plus+=1
                else:
                    minus+=1
            if plus>minus:
                if _clashfree_annot_pos(annot_pos_p, coords):
                    ax.annotate(annot_dict[iloop], xy=annot_pos_p,
                                ha="center", va="center", **text_kwargs )
                else:
                    log.info("Cannot annotate %s as '%s' (only trying inside and right side), because of insufficient space.", iloop, annot_dict[iloop])

            else:
                if _clashfree_annot_pos(annot_pos_m, coords):
                    ax.annotate(annot_dict[iloop], xy=annot_pos_m,
                                ha="center", va="center", **text_kwargs )
                else:
                    log.info("Cannot annotate %s as '%s' (only trying inside and left side), because of insufficient space.", iloop, annot_dict[iloop])
    for mloop in itertools.chain(cg.floop_iterator(), cg.tloop_iterator(), cg.mloop_iterator()):
        nt1, nt2 = cg.define_a(mloop)
        res = list(cg.define_residue_num_iterator(mloop))
        if len(res)==0:
            anchor = np.mean([coords[nt1-1], coords[nt2-1]], axis=0)
        elif len(res)%2==1:
            anchor = coords[res[int(len(res)//2)]-1]
        else:
            anchor =  np.mean([ coords[res[int(len(res)//2)-1]-1],
                                coords[res[int(len(res)//2)]-1] ],
                              axis=0)
        loop_vec = coords[nt1-1] - coords[nt2-1]
        norm_vec = (loop_vec[1], -loop_vec[0])
        norm_vec/=ftuv.magnitude(norm_vec)
        annot_pos = anchor - norm_vec*18
        if _clashfree_annot_pos(annot_pos, coords):
            ax.annotate(annot_dict[mloop], xy=annot_pos,
                        ha="center", va="center", **text_kwargs )
        else:
            log.info("Cannot annotate %s as '%s' , because of insufficient space.", mloop, annot_dict[mloop])

def plot_rna_multiple(cg, separator_positions = None, ax=None, ignore_gaps = True, offset=(0, 0), text_kwargs={}, backbone_kwargs={},
             basepair_kwargs={}, color=True, color_method = "standard", lighten=0, annotations={}, custom_colors=None):
    '''
    Plot a multi-strand RNA structure given a set of nucleotide coordinates

    .. note::

        This function calls set_axis_off on the axis. You can revert this by
        using ax.set_axis_on() if you like to see the axis.

    :param cg: A forgi.threedee.model.coarse_grain.CoarseGrainRNA structure
    :param separator_positions: A list of gap positions to separate different strands.
    :param color_method: Specify the color method
                   --- "standard" : default method
                   --- "strands" : color strands differently
                   --- "custom" : color nucleotides individually
    :param ax: A matplotlib plotting area
    :param offset: Offset the plot by these coordinates. If a simple True is passed in, then
                   offset by the current width of the plot
    :param text_kwargs: keyword arguments passed to matplotlib.pyplot.annotate
                        for plotting of the sequence
    :param backbone_kwargs: keyword arguments passed to matplotlib.pyplot.plot
                        for plotting of the backbone links
    :param basepair_kwargs: keyword arguments passed to matplotlib.pyplot.plot
                        for plotting of the basepair links
    :param lighten: Make circles lighter. A percent value where 1 makes
                    everything white and 0 leaves the colors unchanged
    :param annotations: A dictionary {elem_name: string} or None.
                        By default, the element names (e.g. "s0") are plotted
                        next to the element. This dictionary can be used to
                        override the default element names by costum strings.
                        To remove individual annotations, assign an empty string to the key.
                        To remove all annotations, set this to None.

                        .. warning::

                            Annotations are not shown, if there is not enough space.
                            Annotations not shown are logged with level INFO
    :return: (ax, coords) The axes and the coordinates for each nucleotide
    '''
    log.info("Starting to plot RNA...")
    import RNA
    import matplotlib.colors as mc
    RNA.cvar.rna_plot_type = 1

    coords = []
    #colors = []
    #circles = []

    bp_string = cg.to_dotbracket_string()
    # get the type of element of each nucleotide
    el_string = cg.to_element_string()
    # i.e. eeesssshhhhsssseeee
    el_to_color = {'f': 'orange',
                   't': 'orange',
                   's': 'green',
                   'h': 'blue',
                   'i': 'yellow',
                   'm': 'red'}

    if ax is None:
        ax = plt.gca()

    if offset is None:
        offset = (0, 0)
    elif offset is True:
        offset = (ax.get_xlim()[1], ax.get_ylim()[1])
    else:
        pass

    vrna_coords = RNA.get_xy_coordinates(bp_string)

    # TODO Add option to rotate the plot
    for i, _ in enumerate(bp_string):
        coord = (offset[0] + vrna_coords.get(i).X,
                 offset[1] + vrna_coords.get(i).Y)
        coords.append(coord)
    coords = np.array(coords)

    # First plot backbone
    bkwargs = {"color":"black", "zorder":0}
    bkwargs.update(backbone_kwargs)

    
    # Define the list of Segments
    segment_list = []
    prev_idx = 0

    if separator_positions:
        for i, idx in enumerate(separator_positions):
            if i == 0:
                segment_list.append((0, idx-1))
                prev_idx = idx
            elif i == len(separator_positions)-1:
                segment_list.append((idx+1, len(coords)-1))
            elif idx - prev_idx > 1:
                segment_list.append((prev_idx+1, idx-1))
                prev_idx = idx
            else:
                prev_idx = idx

    # Backbone
    # Skip the connection indicated by separator_positions and draw only the segments
    if ignore_gaps:
        for segment in segment_list:
            segment_coords = coords[segment[0]:segment[1]+1]
            ax.plot(segment_coords[:, 0], segment_coords[:, 1], **bkwargs)
    else:
        ax.plot(coords[:, 0], coords[:, 1], **bkwargs)

    # Now plot basepairs
    basepairs = []
    for s in cg.stem_iterator():
        for p1, p2 in cg.stem_bp_iterator(s):
            basepairs.append([coords[p1-1], coords[p2-1]])
    if basepairs:
        basepairs = np.array(basepairs)
        if color:
            c = "red"
        else:
            c = "black"
        bpkwargs = {"color":c, "zorder":0, "linewidth":3}
        bpkwargs.update(basepair_kwargs)
        ax.plot(basepairs[:,:,0].T, basepairs[:,:,1].T, **bpkwargs)


    # Now plot circles
    if color_method == "standard":
        for i, coord in enumerate(coords):
            # if i in separator_positions:
            #     continue
            if color:
                c = el_to_color[el_string[i]]
                h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))
                if lighten>0:
                    l += (1-l)*min(1,lighten)
                else:
                    l +=l*max(-1, lighten)
                if l>1 or l<0:
                    print(l)
                c=colorsys.hls_to_rgb(h,l,s)
                circle = plt.Circle((coord[0], coord[1]),
                                color=c)
            else:
                circle = plt.Circle((coord[0], coord[1]),
                                    edgecolor="black", facecolor="white")

            ax.add_artist(circle)
            if cg.seq:
                if "fontweight" not in text_kwargs:
                    text_kwargs["fontweight"]="bold"
                ax.annotate(cg.seq[i+1],xy=coord, ha="center", va="center", **text_kwargs)

    elif color_method == "custom":
        if custom_colors is None:
            raise ValueError("Must supply a custom_colors dictionary with color_method='custom'")

        for i, coord in enumerate(coords):
            c = custom_colors.get(i, "#FFFFFF")  # default to white or some fallback
            h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
            l = min(1, l + (1 - l) * lighten) if lighten >= 0 else max(0, l * (1 + lighten))
            adjusted_color = colorsys.hls_to_rgb(h, l, s)

            circle = plt.Circle((coord[0], coord[1]), color=adjusted_color)
            ax.add_artist(circle)

            if cg.seq:
                label = cg.seq[i + 1]  # 1-indexed
                if "fontweight" not in text_kwargs:
                    text_kwargs["fontweight"] = "bold"
                ax.annotate(label, xy=coords[i], ha="center", va="center", **text_kwargs)

    elif color_method == "strands":
        strand_colors = get_strand_colors(len(segment_list))

        offset = 0
        for strand_idx in range(len(segment_list)):
            start = segment_list[strand_idx][0]
            end = segment_list[strand_idx][1]+1

            base_color = strand_colors[strand_idx]
            h, l, s = colorsys.rgb_to_hls(*base_color)
            l = min(1, l + (1 - l) * lighten) if lighten >= 0 else max(0, l * (1 + lighten))
            adjusted_color = colorsys.hls_to_rgb(h, l, s)

            for i in range(start, end):
                circle = plt.Circle((coords[i][0], coords[i][1]), color=adjusted_color)
                ax.add_artist(circle)

                if cg.seq:
                    label = cg.seq[i + 1]  # 1-indexed
                    if "fontweight" not in text_kwargs:
                        text_kwargs["fontweight"] = "bold"
                    ax.annotate(label, xy=coords[i], ha="center", va="center", **text_kwargs)

    else:
        raise ValueError("Unknown color method")

    
    if ignore_gaps:
        # Remove the separators from coords and ch
        mask = np.ones(len(coords), dtype=bool)
        mask[separator_positions] = False
        coords = coords[mask]

        dotbracket = cg.to_dotbracket_string()
        seq = str(cg.seq)
        dotbracket = [dotbracket[i] for i in range(len(dotbracket)) if i not in separator_positions]
        seq = [seq[i] for i in range(len(seq)) if i not in separator_positions]
        dotbracket = "".join(dotbracket)
        seq = "".join(seq)
        cg = BulgeGraph.from_dotbracket(dotbracket,  seq)

        all_coords= list(coords)
        ntnum_kwargs = {"color":"gray"}
        ntnum_kwargs.update(text_kwargs)
        for nt in range(10, cg.seq_length, 10):
            # We try different angles
            annot_pos = _find_annot_pos_on_circle(nt, all_coords, cg)
            if annot_pos is not None:
                ax.annotate(str(nt), xy=coords[nt-1], xytext=annot_pos,
                            arrowprops={"width":1, "headwidth":1, "color":"gray"},
                            ha="center", va="center", zorder=0, **ntnum_kwargs)
                all_coords.append(annot_pos)


        _annotate_rna_plot(ax, cg, all_coords, annotations, text_kwargs)
    else:
        all_coords= list(coords)
        ntnum_kwargs = {"color":"gray"}
        ntnum_kwargs.update(text_kwargs)
        for nt in range(10, cg.seq_length, 10):
            # We try different angles
            annot_pos = _find_annot_pos_on_circle(nt, all_coords, cg)
            if annot_pos is not None:
                ax.annotate(str(nt), xy=coords[nt-1], xytext=annot_pos,
                            arrowprops={"width":1, "headwidth":1, "color":"gray"},
                            ha="center", va="center", zorder=0, **ntnum_kwargs)
                all_coords.append(annot_pos)


        _annotate_rna_plot(ax, cg, all_coords, annotations, text_kwargs)


    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),
                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),
               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),
                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))

    '''
    min_coord = min(datalim[0][0], datalim[0][1])
    max_coord = max(datalim[1][0], datalim[1][1])
    datalim = ((min_coord, min_coord), (max_coord, max_coord))

    print "min_coord:", min_coord
    print "max_coord:", max_coord
    print "datalime:", datalim
    '''

    width = datalim[1][0] - datalim[0][0]
    height = datalim[1][1] - datalim[0][1]

    #ax.set_aspect(width / height)
    ax.set_aspect('equal', 'datalim')
    ax.update_datalim(datalim)
    ax.autoscale_view()
    ax.set_axis_off()

    return (ax, coords)


def _find_annot_pos_on_circle(nt, coords, cg):
    for i in range(5):
        for sign in [-1,1]:
            a = np.pi/4*i*sign
            if cg.get_elem(nt)[0]=="s":
                bp = cg.pairing_partner(nt)
                anchor = coords[bp-1]
            else:
                anchor =np.mean([ coords[nt-2], coords[nt]], axis=0)
            vec = coords[nt-1]-anchor
            vec=vec/ftuv.magnitude(vec)
            rotated_vec =  np.array([vec[0]*math.cos(a)-vec[1]*math.sin(a),
                                     vec[0]*math.sin(a)+vec[1]*math.cos(a)])
            annot_pos = coords[nt-1]+rotated_vec*18
            if _clashfree_annot_pos(annot_pos, coords):
                log.debug("Annot pos on c is %s",annot_pos)
                return annot_pos
    return None
