import os
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import m8r
from PIL import Image
import torch

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def visVelocityprofile(velprofile, path, name, x0, dx, t0, dt, figsize = cm2inch(14,16), vmin = None, vmax = None, fontsize = 20, labelsize = 20, xlabelname = 'CDP', ylabelname = 'Time (s)', cmap = 'jet', srcorrlist = None, axis_off = False, demicals = '%.1f', title = None):
    if path is not None and name is not None:
        print(f" >>> save at {path}/{name}")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    nt, nx = velprofile.shape
    t = np.arange(0, nt) * dt + t0
    xrange = [x0, x0 + nx * dx]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        velprofile,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=[xrange[0], xrange[-1], t[-1], t[0]],
    )
    ax.set_xlabel(xlabelname, fontname="Times New Roman", fontsize=fontsize)
    ax.set_ylabel(ylabelname, fontname="Times New Roman", fontsize=fontsize)
    ax.xaxis.tick_top()  # Place ticks on top
    ax.xaxis.set_label_position("top")  # Place label on top
    ax.tick_params(axis="both", labelsize=labelsize)
    if srcorrlist is not None:
        for srcorr in srcorrlist:
            ax.scatter(
                srcorr[0],
                np.zeros_like(srcorr[0]),
                color=srcorr[1],
                s=srcorr[4],
                marker=srcorr[2],
                label=srcorr[3],
            )
        ax.legend(loc="upper right", fontsize=labelsize)

    if axis_off:
        plt.axis("off")
    else:
        # if showcbar:
        fig.subplots_adjust(right=0.95)
        position = fig.add_axes([0.96, 0.1, 0.015, 0.8])  # Position [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=position)
        cbar.ax.tick_params(labelsize=labelsize)
        # set cbar format
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(demicals))
        # set cbar fontname
        cbar.ax.yaxis.label.set_fontname("Times New Roman")

    if title is not None:
        ax.set_title(title, fontname="Times New Roman", fontsize=fontsize)

    if path is not None and name is not None:
        plt.savefig(
            os.path.join(path, name), dpi=300, bbox_inches="tight", pad_inches=0.0
        )
    plt.close()


def visprofile(
    profile,
    path,
    name,
    t0,
    dt,
    x0,
    dx,
    size=(251, 251),
    figsize=cm2inch(14, 16),
    agc=False,
    vmin=None,
    vmax=None,
    fontsize=20,
    labelsize=20,
    cmap="seismic",
    demicals="%.1f",
    axisoff=False,
    showcbar=True,
    PlotLineAtCDPLoc=[],
    ylabelname="Time (s)",
    xlabelname="CDP",
    rect_region=None,
    fontname="Arial",
):
    print(f">>> save at {path}/{name}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if agc:
        print(f">>> agc ...")
        profile, _ = enhance(profile, size)
    if vmin is None:
        vmin = -2 * (std_val := np.std(profile))
        vmax = 2 * std_val

    nt, nx = profile.shape
    t = np.arange(0, nt) * dt + t0
    x = np.arange(0, nx) * dx + x0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        profile,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=[x[0], x[-1], t[-1], t[0]],
    )
    ax.set_ylabel(ylabelname, fontsize=fontsize, fontname=fontname)
    ax.set_xlabel(xlabelname, fontsize=fontsize, fontname=fontname)

    ax.tick_params(axis="both", labelsize=labelsize)

    ax.xaxis.tick_top()  # Place ticks on top
    ax.xaxis.set_label_position("top")  # Place label on top
    if axisoff:
        plt.axis("off")

    if showcbar:
        fig.subplots_adjust(right=0.95)
        position = fig.add_axes([0.96, 0.1, 0.015, 0.8])  # Position [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=position)
        cbar.ax.tick_params(labelsize=labelsize)
        # set cbar format
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter(demicals))
        # set cbar fontname
        cbar.ax.yaxis.label.set_fontname(fontname)

    if PlotLineAtCDPLoc is not None:
        if len(PlotLineAtCDPLoc) > 0:
            for cdp_loc in PlotLineAtCDPLoc:
                ax.axvline(x=cdp_loc, color="k", linewidth=2, linestyle="--", alpha=0.5)
    if rect_region is not None:
        basename = os.path.basename(name).split(".")[0]
        for _, region in rect_region.items():
            # for different region, set different edge color
            rect = plt.Rectangle(
                (region["x1"], region["y1"]),
                region["x2"] - region["x1"],
                region["y2"] - region["y1"],
                fill=False,
                linewidth=3,
                linestyle="--",
                edgecolor=region["color"],
            )
            ax.add_patch(rect)
            rectxid1 = np.int32((region["x1"] - x0) // dx)
            rectxid2 = np.int32((region["x2"] - x0) // dx)
            rectyid1 = np.int32((region["y1"] - t0) // dt)
            rectyid2 = np.int32((region["y2"] - t0) // dt)
            rectprofile = profile[rectyid1:rectyid2, rectxid1:rectxid2]
            tcorrrect = t[rectyid1:rectyid2]
            xcorrect = x[rectxid1:rectxid2]
            rectprofile /= np.sqrt(np.mean(np.square(rectprofile)))
            figrect, axrect = plt.subplots()
            axrect.imshow(
                rectprofile,
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[xcorrect[0], xcorrect[-1], tcorrrect[-1], tcorrrect[0]],
            )
            axrect.axis("off")
            rect_border = plt.Rectangle(
                (xcorrect[0], tcorrrect[-1]),  # Bottom-left coordinates
                xcorrect[-1] - xcorrect[0],  # Rectangle width
                tcorrrect[0] - tcorrrect[-1],  # Rectangle height
                fill=False,
                edgecolor=region["color"],  # Border color
                linewidth=6,  # Border width,
                linestyle="--",
            )
            axrect.add_patch(rect_border)
            plt.savefig(
                os.path.join(
                    path,
                    f'{basename}_rect_{region["x1"]}_{region["y1"]}_{region["x2"]}_{region["y2"]}.pdf',
                ),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.0,
            )
            plt.close(figrect)

    plt.savefig(os.path.join(path, name), dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def create_mask(nz, noffset, start_cut_x, start_cut_y, end_cut_x, end_cut_y):
    mask_zeros = np.zeros((nz, noffset))
    k = (end_cut_y - start_cut_y) / (end_cut_x - start_cut_x)
    b = start_cut_y - k * start_cut_x
    for xid in range(noffset):
        yid = np.max(np.round(k * xid + b).astype(np.int32), 0)
        mask_zeros[yid:,xid] = 1
    return mask_zeros

def load_networks(model, device, load_path = None, continue_train = False):
    """Load all the networks from the disk.

    Parameters:
        model (torch.nn.Module) -- the network to be loaded
        load_path (str) -- the path to load the network from; if None, will use the name saved in <model_names>
        continue_train (bool) -- if continue training, the network will be initialized with the lastest checkpoint.
    """

    state_dict = torch.load(load_path, map_location=device)

    if continue_train:
        net_statedic = model.state_dict()
        load_state_dic = {}
        for k, v in state_dict.items():
          if 'module' in k:
            if isinstance(model, torch.nn.DataParallel) and net_statedic[k].shape == v.shape:
              load_state_dic[k] = v
            else:
              load_state_dic[k.replace('module.', '')] = v
          else:
            if isinstance(model, torch.nn.DataParallel) and net_statedic['module.'+k].shape == v.shape:
              load_state_dic['module.'+k] = v
            else:
              load_state_dic[k] = v
        print(f'found {len(load_state_dic)} keys in the checkpoint')
        net_statedic.update(load_state_dic)
        model.load_state_dict(net_statedic)
    else:
        if isinstance(model, torch.nn.DataParallel):
          model.module.load_state_dict(state_dict)
        else:
          model.load_state_dict(state_dict)    
    return model

def imresize(input_array, out_shape, mode = 'bilinear'):
    # input_array: 2d array
    # out_shape: tuple
    # mode: bilinear or nearest
    resample_mode = Image.BILINEAR if mode == 'bilinear' else Image.NEAREST
    input_array = Image.fromarray(input_array)
    input_array = input_array.resize(out_shape, resample = resample_mode)
    input_array = np.array(input_array)
    return input_array

def get_axis(File,axis):
    o = File.float("o%d"%axis)
    d = File.float("d%d"%axis)
    n = File.int("n%d"%axis)
    return o, d, n 


def put_axis(File,axis,o,d,n):
    File.put("o%d"%axis,o*1.0)
    File.put("d%d"%axis,d*1.0)
    File.put("n%d"%axis,int(n))
    return o, d, n 

def loadvel(velpath):
    if velpath.endswith('.npy'):
        vel = np.load(velpath)
    elif velpath.endswith('.rsf'):
        vel, _, _ = ReadRSFFile(velpath, ndims = 2)
        vel = vel.T
    else:
        raise ValueError('velpath should be .npy or .rsf file')
    return vel

def ReadRSFFile(rsfile, ndims = 3):
    '''
    Read RSF file
    Parameters:
        rsfile: RSF file name, Shot gather file
        ndims: dimension of the RSF file
    '''
    shotfile = m8r.Input(rsfile)
    nX = np.zeros(ndims, dtype = np.int32)
    oX = np.zeros(ndims, dtype = np.float32)
    dX = np.zeros(ndims, dtype = np.float32)
    for ndim in range(ndims):
        oX[ndim], dX[ndim], nX[ndim] = get_axis(shotfile, ndim+1)
    oX.tolist().reverse()
    dX.tolist().reverse()
    shot = np.zeros(np.prod(nX), dtype = np.float32)
    shot = shotfile.read(shape = nX.tolist().reverse())
    shotfile.close()
    return shot, oX, dX

def convertdata2rsf(data, o_data, d_data, fileout, unit = None):
    '''
    Transform numpy array to rsf file
    '''
    Fout = m8r.Output(fileout)
    dim = 1
    for nd, od, dd in zip(reversed(data.shape), reversed(o_data), reversed(d_data)):
        put_axis(Fout, dim, od, dd, nd)
        dim += 1
    if unit is not None:
        assert len(unit) == len(o_data)
        for unit_i, dim_i in zip(unit, range(1, len(unit)+1)):
            print('unit%d'%dim_i, unit_i)
            Fout.put('unit%d'%dim_i, unit_i)
            print('end')
    Fout.write(data)
    Fout.close()

def cm2inch(*tupl):
  inch = 2.54
  if isinstance(tupl[0], tuple):
      return tuple(i/inch for i in tupl[0])
  else:
      return tuple(i/inch for i in tupl)

def enhance(profile, size=(251, 251)):
  mask  = scipy.ndimage.uniform_filter(np.abs(profile), size=(5, 5)) < 1.0e-3
  scale = np.square(profile)
  scale = scipy.ndimage.uniform_filter(scale, size=size, output=scale)
  np.sqrt(scale, out=scale)
  scale[mask] = 1.0
  profile /= scale
  profile[mask] = 0.0
  return profile, scale
