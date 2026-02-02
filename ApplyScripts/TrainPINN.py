# %%
import os
import sys

import json
import numpy as np
from torch.autograd import grad
import torch
from loguru import logger
import pytorch_warmup as warmup
import os
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from util import skplot
from model import networks

    
def update_learningrate(optimizers, schedulers, logger, lr_policy="warm_up"):
    old_lr = optimizers[0].param_groups[0]["lr"]
    for scheduler in schedulers:
        if lr_policy == "plateau":
            scheduler.step(0)
        elif lr_policy == "warm_up":
            with scheduler[1].dampening():
                scheduler[0].step()
        else:
            scheduler.step()
    lr = optimizers[0].param_groups[0]["lr"]
    logger.info(f"learning rate %.7f -> %.7f" % (old_lr, lr))
    return lr


def caculateT0_px_pz_sid(
    velmodel,
    SRX,
    SRZ,
    X,
    Z,
    zmin,
    xmin,
    dz,
    dx,
    TimetableSource,
    srcxorr,
    srz=0,
    TOLX=0.006,
    TOLZ=0.004,
):
    nz, nx, nsrc = SRX.shape
    X_star = [Z.reshape(-1, 1), X.reshape(-1, 1), SRX.reshape(-1, 1)]
    vs = velmodel[
        np.round((SRZ - zmin) / dz).astype(int),
        np.round((SRX - xmin) / dx).astype(int),
        0,
    ]
    T0 = np.sqrt((X - SRX) ** 2 + (Z - SRZ) ** 2) / vs
    px0 = np.divide(X - SRX, T0 * (vs**2), out=np.zeros_like(T0), where=T0 != 0)
    pz0 = np.divide(Z - SRZ, T0 * (vs**2), out=np.zeros_like(T0), where=T0 != 0)
    sids = np.array([]).astype(int)
    # find source location id in X_starf
    for sxi in srcxorr:
        sid, _ = np.where(
            np.logical_and(
                np.logical_and(
                    np.abs(X_star[0] - srz) < TOLZ, np.abs(X_star[1] - sxi) < TOLX
                ),
                np.abs(X_star[2] - sxi) < TOLX,
            )
        )
        sids = np.append(sids, sid)

    leftid = np.arange(len(X_star[0]))
    leftid = np.delete(leftid, sids)

    TimetableSource = TimetableSource.reshape(-1, 1)
    T0 = T0.reshape(-1, 1)

    TimetableSource[sids] = 1
    TimetableSource[leftid] = TimetableSource[leftid] / T0[leftid]

    TimetableSource = TimetableSource.reshape(nz, nx, nsrc)
    T0 = T0.reshape(nz, nx, nsrc)

    return T0, px0, pz0, TimetableSource


def DVGPDETomoGradientEnhancedLoss(
    time_src,
    velupdate,
    zcorr,
    xcorr,
    t0src,
    px0src,
    pz0src,
    lossfn,
    timetabledx,
    timetabledz,
    vminvel=None,
    vmaxvel=None,
):
    dt_dz = grad(
        time_src,
        zcorr,
        grad_outputs=torch.ones_like(time_src),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dt_dx = grad(
        time_src,
        xcorr,
        grad_outputs=torch.ones_like(time_src),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    cond = 1 / (velupdate**2)

    rz = t0src * dt_dz + pz0src * time_src
    rx = t0src * dt_dx + px0src * time_src
    
    pde_time_src_loss = torch.sqrt(
        lossfn(
            rz ** 2 + rx ** 2,
            cond,
        )
    )

    velpdeoutput = 1 / (
        torch.sqrt(rz ** 2 + rx ** 2) + 1e-8
    )

    pde_time_src_loss = pde_time_src_loss + torch.sqrt(lossfn(velupdate, velpdeoutput))

    dz_loss = torch.sqrt(lossfn(rz, timetabledz))

    dx_loss = torch.sqrt(lossfn(rx, timetabledx))

    graconstrain = 1 / (torch.sqrt(timetabledz**2 + timetabledx**2) + 1e-8)
    if vminvel is not None and vmaxvel is not None:
        graconstrain = torch.clamp(graconstrain, vminvel, vmaxvel)
    veloss = torch.sqrt(lossfn(velupdate, graconstrain))

    return pde_time_src_loss, dz_loss, dx_loss, veloss


def loadvel(velpath):
    if velpath.endswith(".npy"):
        vel = np.load(velpath)
    elif velpath.endswith(".rsf"):
        vel, _, _ = skplot.ReadRSFFile(velpath, ndims=2)
        vel = vel.T
    else:
        raise ValueError("velpath should be .npy or .rsf file")
    return vel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PINN Python code")
    parser.add_argument(
        "--initvelpath", type=str, help="Path to the init velocity file", default=None, required=True
    )
    parser.add_argument(
        "--inittimetablepath",
        type=str,
        help="Path to the init timetable file",
        default=None,
        required=True,
    )
    parser.add_argument("--expiter", type=int, help="exp iter", default=0)
    parser.add_argument("--config_file", type=str, help="Path to the config file", required=True)
    parser.add_argument(
        "--truevelpath", type=str, help="Path to the true velocity file", default=None
    )
    parser.add_argument(
        "--SaveEpoch", type=int, nargs="+", help="Epochs to save", default=[]
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    Expname = config["Expname"]
    if not os.path.exists("./result/%s" % Expname):
        os.makedirs("./result/%s" % Expname)
    basefilename = config["basefilename"] + f"_exp{args.expiter}"

    exppath = os.path.join("./result", Expname, basefilename)

    os.makedirs(f"./result/{Expname}/backup", exist_ok=True)
    backup_dir = f"./result/{Expname}/backup"
    os.system("cp ApplyScripts/*.py %s/" % backup_dir)
    os.system("cp model/*.py %s" % backup_dir)
    os.system("cp util/*.py %s" % backup_dir)
    os.system('cp %s %s/' % (args.config_file, backup_dir))

    logger.add(
        os.path.join(exppath, "PINNTomo.log"),
        rotation="500 MB",
        compression="zip",
        level="INFO",
    )
    GPUIDS = config["GPUIDs"]
    initvelpath = args.initvelpath
    inittimetablepath = args.inittimetablepath
    truevelpath = args.truevelpath
    savedepoch = args.SaveEpoch

    # Step1: set observation System and load data
    xmin = config["o_cmp"]
    ncmp = config["ncmp"]
    dx = config["d_cmp"]
    x = np.arange(ncmp) * dx + xmin
    xmax = x[-1]
    zmin = config["oz"]
    nz = config["nz"]
    dz = config["dz"]
    z = np.arange(nz) * dz + zmin
    zmax = z[-1]
    # srxbeg = config['srxbeg']; dsrx = config['dsrx']; nsrx = config['nsrx']; srcxorr = np.arange(srxbeg, srxbeg+nsrx*dsrx, dsrx)
    srxbeg = config["o_cmp"]
    dsrx = config["d_cmp"]
    nsrx = config["ncmp"]
    srcxorr = np.arange(srxbeg, srxbeg + nsrx * dsrx, dsrx)

    subsrcsample = config["subsrcsample"]
    srz = config["srz"]

    srcxorr = srcxorr[::subsrcsample]

    torch.cuda.set_device("cuda:{}".format(GPUIDS[0]))
    ZGRID, XGRID = np.meshgrid(z, x, indexing="ij")

    initvel = loadvel(initvelpath)
    initvelTest = initvel.copy()
    initvelTestOrishape = initvelTest.copy()
    truevel = None
    if truevelpath is not None:
        truevel = loadvel(truevelpath)
        trueloss = []
        trueepoch = []
        trueloss.append(np.sqrt(np.mean((initvel - truevel) ** 2)))
        trueepoch.append(0)

    vminvel = config["velmin"]
    vmaxvel = config["velmax"]

    skplot.visVelocityprofile(
        initvel,
        os.path.join(exppath, "ApplyFigure"),
        f"initvel.png",
        xmin,
        dx,
        zmin,
        dz,
        figsize=skplot.cm2inch(20, 7.5),
        fontsize=10,
        labelsize=10,
        vmin=vminvel,
        vmax=vmaxvel,
        xlabelname="Lateral (km)",
        ylabelname="Depth (km)",
    )

    if inittimetablepath.endswith(".npy"):
        TimetableSourceTrue = np.load(inittimetablepath)
    elif inittimetablepath.endswith(".rsf"):
        TimetableSourceTrue, _, _ = skplot.ReadRSFFile(inittimetablepath, ndims=3)
        TimetableSourceTrue = np.ascontiguousarray(
            np.transpose(TimetableSourceTrue, (2, 1, 0))
        )  # nz, nx, nshot
    else:
        raise ValueError("inittimetablepath should be .npy or .rsf file")

    TimetableSourceTrue = TimetableSourceTrue[:, :, ::subsrcsample]
    TimetableSourceTrueDx = (
        TimetableSourceTrue[:, 2:, :] - TimetableSourceTrue[:, :-2, :]
    ) / (2 * dx)
    TimetableSourceTrueDz = (
        TimetableSourceTrue[2:, :, :] - TimetableSourceTrue[:-2, :, :]
    ) / (2 * dz)

    TimetableSourceTrueDx = np.pad(
        TimetableSourceTrueDx, ((0, 0), (1, 1), (0, 0)), "edge"
    )
    TimetableSourceTrueDz = np.pad(
        TimetableSourceTrueDz, ((1, 1), (0, 0), (0, 0)), "edge"
    )
    
    device = torch.device("cuda:{}".format(GPUIDS[0]))

    initvel = np.repeat(initvel[..., np.newaxis], srcxorr.size, axis=2)

    Z, X, SRX = np.meshgrid(
        z, x, srcxorr, indexing="ij"
    )  # Z: depth, X: distance, SRX: source location shape: (nz, ncmp, nsrc)
    SRZ = np.ones(SRX.shape) * srz
    TimetableSourceprocess = TimetableSourceTrue.copy()
    T0, px0, pz0, TimetableSourceprocess = caculateT0_px_pz_sid(
        initvel,
        SRX,
        SRZ,
        X,
        Z,
        zmin,
        xmin,
        dz,
        dx,
        TimetableSourceprocess,
        srcxorr,
        srz=srz,
    )

    Z = Z.reshape(-1, 1)
    X = X.reshape(-1, 1)
    SRX = SRX.reshape(-1, 1)
    T0 = T0.reshape(-1, 1)
    px0 = px0.reshape(-1, 1)
    pz0 = pz0.reshape(-1, 1)
    TimetableSourceprocess = TimetableSourceprocess.reshape(-1, 1)
    TimetableSourceTrueDx = TimetableSourceTrueDx.reshape(-1, 1)
    TimetableSourceTrueDz = TimetableSourceTrueDz.reshape(-1, 1)
    initvel = initvel.reshape(-1, 1)

    ZGRID, XGRID = np.meshgrid(z, x, indexing="ij")

    ZGRID = ZGRID.reshape(-1, 1)
    XGRID = XGRID.reshape(-1, 1)
    initvelTest = initvelTest.reshape(-1, 1)

    tomo_model = networks.PINN(
        n1ForTime=10,
        n1ForVel=10,
        input_num_for_time=6,
        input_num_for_vel=2,
        scaleTime=2,
        scaleVel=0.5,
    )

    tomo_model = tomo_model.to(device)

    if len(GPUIDS) > 1:
        logger.info(">>> use multi-GPU training ...")
        tomo_model = torch.nn.DataParallel(
            tomo_model, device_ids=GPUIDS, output_device=GPUIDS[0]
        )

    if config["continueTrain"]:
        logger.info(
            f'>>> continue training, load model from {config["continueTrainModelPath"]} ...'
        )
        tomo_model = skplot.load_networks(
            tomo_model,
            device,
            config["continueTrainModelPath"],
            continue_train=config["continueTrain"],
        )

    mseLoss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, tomo_model.parameters()),
        lr=config["lr"],
        betas=(0.9, 0.999),
    )

    # Step3: training
    totalTrainpoints = Z.size
    epochs = config["epochs"]

    bestTestLoss = np.inf
    valinteral = config["valinterval"]

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - config["epoch_count"]
    )
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    schedulers = [lr_scheduler, warmup_scheduler]

    totalTestpoints = nz * ncmp

    trainsrcsamplenum = 2
    velLossceoff = config["velLossceoff"]
    data_coeff = config["data_coeff"]
    pde_coeff = config["pde_coeff"]
    batchsize = config["batchsize"] * len(GPUIDS)

    savepath = os.path.join("result", Expname)
    writer = SummaryWriter(os.path.join(exppath, "tb_runs"))
    logger.info(f">>> start training ... {epochs} epochs in total\t ")
    total_parameters = sum(p.numel() for p in tomo_model.parameters())
    # logger total parameters in mb
    logger.info(f"total parameters: {total_parameters * 32 / 8 / 1024 / 1024} MB")
    TotalTrainpoints = len(Z)
    nbatch_Train = (TotalTrainpoints + batchsize - 1) // batchsize
    nbatch_Test = (nz * ncmp + batchsize * len(GPUIDS) - 1) // (batchsize * len(GPUIDS))
    totalTestpoints = nz * ncmp
    logger.info(
        f"Total Train points: {TotalTrainpoints}, Total Test points: {totalTestpoints}\tbatch size: {batchsize}, nbatch_Train: {nbatch_Train}, nbatch_Test: {nbatch_Test}"
    )

    for epoch in range(config["epoch_count"], epochs):
        totalloss_Train = []
        total_data_loss_Train = []
        total_pde_loss_Train = []
        pde_time_src_loss_Train = []
        dz_loss_Train = []
        dx_loss_Train = []
        veloss_Train = []
        optimizer.zero_grad()
        for ibatch in range(nbatch_Train):
            batch_start = ibatch * batchsize
            batch_end = min((ibatch + 1) * batchsize, TotalTrainpoints)
            if batch_end <= batch_start:
                continue
            zcorr = torch.from_numpy(Z[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
            xcorr = torch.from_numpy(X[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
            srcxcorrtrain = torch.from_numpy(SRX[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
            t0train = torch.from_numpy(T0[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
            px0train = torch.from_numpy(px0[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
            pz0train = torch.from_numpy(pz0[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)

            labelTimeTable = (
                torch.from_numpy(TimetableSourceprocess[batch_start:batch_end, :])
                .float()
                .cuda(non_blocking=True)
            )
            initvelmodeltrain = (
                torch.from_numpy(initvel[batch_start:batch_end, :])
                .float()
                .cuda(non_blocking=True)
            )
            timesourcedx = (
                torch.from_numpy(TimetableSourceTrueDx[batch_start:batch_end, :])
                .float()
                .cuda(non_blocking=True)
            )
            timesourcedz = (
                torch.from_numpy(TimetableSourceTrueDz[batch_start:batch_end, :])
                .float()
                .cuda(non_blocking=True)
            )

            timesrcinput = torch.cat(
                (zcorr, xcorr, srcxcorrtrain, t0train, px0train, pz0train), dim=1
            )
            velinput = torch.cat((zcorr, xcorr), dim=1)

            time_src, vel_update = tomo_model(timesrcinput, velinput)
            timedata_loss = torch.sqrt(mseLoss(time_src, labelTimeTable))
            vel_update = initvelmodeltrain + vel_update
            pde_time_src_loss, dz_loss, dx_loss, veloss = (
                DVGPDETomoGradientEnhancedLoss(
                    time_src,
                    vel_update,
                    zcorr,
                    xcorr,
                    t0train,
                    px0train,
                    pz0train,
                    mseLoss,
                    timesourcedx,
                    timesourcedz,
                    vminvel=vminvel,
                    vmaxvel=vmaxvel,
                )
            )

            total_pde_loss = pde_time_src_loss + dz_loss + dx_loss
            loss = (
                timedata_loss * data_coeff
                + total_pde_loss * pde_coeff
                + veloss * velLossceoff
            )

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.error(f"loss is nan or inf")
                raise ValueError("loss is nan or inf")
            loss.backward()
            totalloss_Train.append(loss.item())
            total_data_loss_Train.append(timedata_loss.item())
            total_pde_loss_Train.append(pde_time_src_loss.item())
            pde_time_src_loss_Train.append(pde_time_src_loss.item())
            dz_loss_Train.append(dz_loss.item())
            dx_loss_Train.append(dx_loss.item())
            veloss_Train.append(veloss.item())
            optimizer.step()
            optimizer.zero_grad()
        totalloss_Train = np.mean(totalloss_Train)
        total_data_loss_Train = np.mean(total_data_loss_Train)
        total_pde_loss_Train = np.mean(total_pde_loss_Train)
        pde_time_src_loss_Train = np.mean(pde_time_src_loss_Train)
        dz_loss_Train = np.mean(dz_loss_Train)
        dx_loss_Train = np.mean(dx_loss_Train)
        veloss_Train = np.mean(veloss_Train)
        learning_rate = update_learningrate(
            [optimizer], [schedulers], logger, lr_policy="warm_up"
        )
        writer.add_scalar("Total Loss", totalloss_Train, epoch)
        writer.add_scalar("Tomography Time Loss", total_data_loss_Train, epoch)
        writer.add_scalar("PDE Loss", total_pde_loss_Train, epoch)
        writer.add_scalar("PDE Time Loss", pde_time_src_loss_Train, epoch)
        writer.add_scalar("PDE dz Loss", dz_loss_Train, epoch)
        writer.add_scalar("PDE dx Loss", dx_loss_Train, epoch)
        writer.add_scalar("Velocity Loss", veloss_Train, epoch)
        writer.add_scalar("Learning Rate", learning_rate, epoch)
        logger.info(
            f">>>> epoch: {epoch}, train loss: {totalloss_Train:.4e}, data loss: {total_data_loss_Train:.4e}, total pde loss: {total_pde_loss_Train:.4e}, pde time loss: {pde_time_src_loss_Train:.4e}, dz loss: {dz_loss_Train:.4e}, dx loss: {dx_loss_Train:.4e}, vel loss: {veloss_Train:.4e}\t lr = {learning_rate:.6e}"
        )

        if epoch % valinteral == 0 or epoch == epochs - 1 or epoch in savedepoch:  # test
            logger.info(">>>> start testing ...")
            if epoch in savedepoch:
                if not os.path.exists(os.path.join(exppath, "model_saved")):
                    os.makedirs(os.path.join(exppath, "model_saved"), exist_ok=True)
                if len(GPUIDS) > 1:
                    torch.save(
                        tomo_model.module.state_dict(),
                        os.path.join(exppath, "model_saved", f"epoch{epoch}.pth"),
                    )
                else:
                    torch.save(
                        tomo_model.state_dict(),
                        os.path.join(exppath, "model_saved", f"epoch{epoch}.pth"),
                    )

            velocity_all = np.zeros((nz * ncmp, 1))
            with torch.no_grad():
                rmseloss = 0
                for ibatch in range(nbatch_Test):
                    batch_start = ibatch * batchsize
                    batch_end = min((ibatch + 1) * batchsize, totalTestpoints)
                    if batch_end <= batch_start:
                        continue
                    zcorr = torch.from_numpy(ZGRID[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
                    xcorr = torch.from_numpy(XGRID[batch_start:batch_end, :]).float().requires_grad_(True).cuda(non_blocking=True)
                    _, velupdate = tomo_model(None, torch.cat((zcorr, xcorr), dim=1))
                    inputveltest = initvelTest[batch_start:batch_end, :]
                    velupdate = velupdate.detach().cpu().numpy() + inputveltest
                    velocity_all[batch_start:batch_end, :] = velupdate

            velocity_all = velocity_all.reshape(nz, ncmp)
            if epoch in savedepoch:
                skplot.visVelocityprofile(
                    velocity_all,
                    os.path.join(exppath, "ApplyFigure"),
                    f"outvel_epoch{epoch}.jpg",
                    xmin,
                    dx,
                    zmin,
                    dz,
                    figsize=skplot.cm2inch(20, 7.5),
                    fontsize=10,
                    labelsize=10,
                    xlabelname="Lateral (km)",
                    ylabelname="Depth (km)",
                )
            if truevel is not None:
                trueloss.append(np.sqrt(np.mean((velocity_all - truevel) ** 2)))
                trueepoch.append(epoch)
                writer.add_scalar("True Velocity Loss", trueloss[-1], epoch)

    writer.close()
    logger.info(">>>> Finish training ...")
    np.save(os.path.join(exppath, f"epoch_outvel.npy"), velocity_all)

    if truevel is not None:
        figure, ax = plt.subplots(1, 1, figsize=skplot.cm2inch(10, 7.5))
        ax.plot(trueepoch, trueloss)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("True velocity MSE Loss")
        figure.savefig(os.path.join(exppath, "ApplyFigure", "TruevelMSELoss.png"))
