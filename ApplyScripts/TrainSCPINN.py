# %%
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from loguru import logger
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
import tqdm
from matplotlib import pyplot as plt
import kornia

# add current working directory for relative imports when running as script
sys.path.append(os.getcwd())
from util import skplot
from model import networks

################################################################################
# Utility helpers
################################################################################
def set_global_seed(seed: int = 2024) -> None:
    """Set random seeds for reproducibility (best effort)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic mode where possible (may impact perf)
    torch.backends.cudnn.deterministic = False  # keep False for speed unless strict reproducibility needed
    torch.backends.cudnn.benchmark = True

class RMSELoss(nn.Module):
    """Root MSE with small epsilon to avoid sqrt(0)."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + 1e-12)

def update_learningrate(optimizers, schedulers, logger, lr_policy: str = 'warm_up') -> float:
    """Update learning rate with support for warmup pair passed as a nested list.

    Expected `schedulers` shape: list where each element is either a scheduler or
    a pair [main_scheduler, warmup_scheduler]. This keeps backward compatibility
    with the previous usage which passed `[ [lr_scheduler, warmup_scheduler] ]`.
    """
    old_lr = optimizers[0].param_groups[0]['lr']
    for sch in schedulers:
        if isinstance(sch, (list, tuple)) and len(sch) == 2 and lr_policy == 'warm_up':
            main_sch, warm_sch = sch
            with warm_sch.dampening():
                main_sch.step()
        else:  # fallback
            if lr_policy == 'plateau':
                sch.step(0)
            else:
                sch.step()
    new_lr = optimizers[0].param_groups[0]['lr']
    logger.info(f'learning rate {old_lr:.7f} -> {new_lr:.7f}')
    return new_lr

def caculateT0_px_pz_sid(velmodel, SRX, SRZ, X, Z, zmin, xmin, dz, dx, TimetableSource, srcxorr, srz = 0, TOLX = 0.006, TOLZ = 0.004):  # noqa: E501
    nz, nx, nsrc = SRX.shape
    X_star = [Z.reshape(-1, 1), X.reshape(-1, 1), SRX.reshape(-1, 1)]
    vs = velmodel[np.round((SRZ - zmin) / dz).astype(int), np.round((SRX - xmin) / dx).astype(int), 0]
    T0 = np.sqrt((X - SRX)**2 + (Z - SRZ)**2) / vs
    px0 = np.divide(X-SRX, T0*(vs**2), out=np.zeros_like(T0), where=T0!=0)
    pz0 = np.divide(Z-SRZ, T0*(vs**2), out=np.zeros_like(T0), where=T0!=0)
    sids = np.array([]).astype(int)
    # find source location id in X_starf
    for sxi in srcxorr:
        sid,_ = np.where(np.logical_and(np.logical_and(np.abs(X_star[0]-srz)<TOLZ , np.abs(X_star[1]-sxi)<TOLX), np.abs(X_star[2]-sxi)<TOLX))
        sids = np.append(sids,sid)

    print(f'len X_STAR: {len(X_star[0])}, len sids: {len(sids)}')
    leftid = np.arange(len(X_star[0]))
    leftid = np.delete(leftid, sids)

    TimetableSource = TimetableSource.reshape(-1, 1)
    T0 = T0.reshape(-1, 1)

    TimetableSource[sids] = 1
    TimetableSource[leftid] = TimetableSource[leftid] / T0[leftid]

    TimetableSource = TimetableSource.reshape(nz, nx, nsrc)
    T0 = T0.reshape(nz, nx, nsrc)

    return T0, px0, pz0, TimetableSource

def DVGPDETomoGradientEnhancedLoss(time_src, velupdate, zcorr, xcorr,
                                   t0src, px0src, pz0src,
                                   rmse_loss, timetabledx, timetabledz,
                                   vminvel=None, vmaxvel=None):
    # Pre-calculate gradients once
    dt_dz = grad(time_src, zcorr,
                 grad_outputs=torch.ones_like(time_src),
                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    dt_dx = grad(time_src, xcorr,
                 grad_outputs=torch.ones_like(time_src),
                 create_graph=True, retain_graph=True, only_inputs=True)[0]

    # PDE residual term
    rz = t0src * dt_dz + pz0src * time_src
    rx = t0src * dt_dx + px0src * time_src
    residual_sq = rz.pow(2) + rx.pow(2)

    inv_v_est = torch.sqrt(residual_sq + 1e-12)  # Equivalent to 1/v
    v_from_pde = 1.0 / (inv_v_est + 1e-8)

    # First type: velocity consistency
    vel_consistency1 = rmse_loss(velupdate, v_from_pde)

    # Second type (based on given gradients)
    inv_v_label = torch.sqrt(timetabledz.pow(2) + timetabledx.pow(2) + 1e-12)
    v_from_label = 1.0 / (inv_v_label + 1e-8)
    if vminvel is not None and vmaxvel is not None:
        v_from_label = torch.clamp(v_from_label, vminvel, vmaxvel)
    vel_consistency2 = rmse_loss(velupdate, v_from_label)

    pde_residual_loss = rmse_loss(residual_sq, (1 / (velupdate.clamp(min=1e-8)**2)))

    dz_loss = rmse_loss(rz, timetabledz)
    dx_loss = rmse_loss(rx, timetabledx)

    return {
        'pde_residual_loss': pde_residual_loss + vel_consistency1 + vel_consistency2,
        'dz_loss': dz_loss,
        'dx_loss': dx_loss,
        'vel_consistency1': vel_consistency1,
        'vel_consistency2': vel_consistency2,
    }

def loadvel(velpath: str) -> np.ndarray:
    if velpath is None:
        raise ValueError('Velocity path is None.')
    if velpath.endswith('.npy'):
        vel = np.load(velpath)
    elif velpath.endswith('.rsf'):
        vel, _, _ = skplot.ReadRSFFile(velpath, ndims=2)
        vel = vel.T
    else:
        raise ValueError('velpath should be .npy or .rsf file')
    return vel

def DipLossWithTV(seismicprofiletensor, outputveltensor, sigma = (0.1, 0.1), kernel_size = 3, epsilon = 1e-8):
    '''
    Calculate the dip loss
    seismicprofiletensor: torch.tensor, shape (N, C, nz, ncmp)
    outputveltensor: torch.tensor, shape (N, C, nz, ncmp)
    '''
    seismicprofiledz = (seismicprofiletensor[:, :, 2:, 1:-1] - seismicprofiletensor[:, :, :-2, 1:-1]) / 2
    seismicprofiledx = (seismicprofiletensor[:, :, 1:-1, 2:] - seismicprofiletensor[:, :, 1:-1, :-2]) / 2
    outputveldz = (outputveltensor[:, :, 2:, 1:-1] - outputveltensor[:, :, :-2, 1:-1]) / 2
    outputveldx = (outputveltensor[:, :, 1:-1, 2:] - outputveltensor[:, :, 1:-1, :-2]) / 2
    tvloss = torch.sqrt(torch.mean(outputveldz ** 2 + outputveldx ** 2))
    seismicprofiledz = kornia.filters.gaussian_blur2d(seismicprofiledz, kernel_size, sigma)
    seismicprofiledx = kornia.filters.gaussian_blur2d(seismicprofiledx, kernel_size, sigma)
    outputveldz = kornia.filters.gaussian_blur2d(outputveldz, kernel_size, sigma)
    outputveldx = kornia.filters.gaussian_blur2d(outputveldx, kernel_size, sigma)
    near_zeros_seismic = seismicprofiledx.abs() < epsilon
    seismicprofiledx[near_zeros_seismic] = epsilon
    near_zeros_vel = outputveldx.abs() < epsilon
    outputveldx[near_zeros_vel] = epsilon
    dipseismic = torch.atan2(seismicprofiledz, seismicprofiledx)
    dipoutput = torch.atan2(outputveldz, outputveldx)
    diploss = torch.sqrt(torch.mean((dipseismic - dipoutput) ** 2))
    return diploss, tvloss

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SI-PINN Python code')
    parser.add_argument('--initvelpath', type=str, help='Path to the init velocity file', default=None)
    parser.add_argument('--inittimetablepath', type=str, help='Path to the init timetable file', default=None)
    parser.add_argument('--stackprofilepath', type=str, help='Path to the stack profile', default=None)
    parser.add_argument('--expiter', type=int, help='exp iter')
    parser.add_argument('--config_file', type=str, help='Path to the config file', required=True)
    parser.add_argument('--truevelpath', type=str, help='Path to the true velocity file', default=None)
    parser.add_argument('--SaveEpoch', type=int, nargs='+', help='Epochs to save', default=[])
    parser.add_argument('--seed', type=int, default=2024, help='Global random seed')
    parser.add_argument('--amp', action='store_true', default=False, help='Enable mixed precision training (torch.cuda.amp)')
    args = parser.parse_args()

    set_global_seed(args.seed)

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    Expname = config['Expname']
    if not os.path.exists('./result/%s' % Expname):
        os.makedirs('./result/%s' % Expname)
    basefilename = config['basefilename'] + f'_exp{args.expiter}'

    exppath = os.path.join('./result', Expname, basefilename)

    os.makedirs(f'./result/{Expname}/backup', exist_ok = True)
    backup_dir = f'./result/{Expname}/backup'
    os.system('cp ApplyScripts/*.py %s/' % backup_dir)
    os.system('cp model/*.py %s' % backup_dir)
    os.system('cp util/*.py %s' % backup_dir)
    # back up config file
    os.system('cp %s %s/' % (args.config_file, backup_dir))

    logger.add(os.path.join(exppath, 'PINNTomo.log'), rotation="500 MB", compression="zip", level="INFO")
    GPUIDS = config['GPUIDs']
    initvelpath = args.initvelpath
    inittimetablepath = args.inittimetablepath
    stackprofilepath = args.stackprofilepath
    truevelpath = args.truevelpath
    savedepoch = args.SaveEpoch

    torch.cuda.set_device('cuda:{}'.format(GPUIDS[0]))
    device = torch.device('cuda:{}'.format(GPUIDS[0]))

    # Step1: set observation System and load data
    xmin = config['o_cmp']; ncmp = config['ncmp']; dx = config['d_cmp']; x = np.arange(ncmp) * dx + xmin; xmax = x[-1]
    zmin = config['oz']; nz = config['nz']; dz = config['dz']; z = np.arange(nz) * dz + zmin; zmax = z[-1]
    srxbeg = config['o_cmp']; dsrx = config['d_cmp']; nsrx = config['ncmp']; srcxorr = np.arange(srxbeg, srxbeg+nsrx*dsrx, dsrx)

    subsrcsample = config['subsrcsample']; srz = config['srz']; DVGModelPath = config["DVGModelPath"]

    srcxorr = srcxorr[::subsrcsample]

    ZGRID, XGRID = np.meshgrid(z, x, indexing='ij')
    initvel = loadvel(initvelpath)
    truevel = loadvel(truevelpath)
    trueloss = []
    trueepoch = []
    trueloss.append(np.sqrt(np.mean((initvel - truevel) ** 2)))
    trueepoch.append(0)

    vminvel = config['velmin']
    vmaxvel = config['velmax']
    print(f'>>> imshow range {vminvel} - {vmaxvel}\t true vel range : {truevel.min()} - {truevel.max()}')

    skplot.visVelocityprofile(initvel, os.path.join(exppath, 'ApplyFigure'), f'initvel.png', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)')

    if inittimetablepath.endswith('.npy'):
        TimetableSourceTrue = np.load(inittimetablepath)
    elif inittimetablepath.endswith('.rsf'):
        TimetableSourceTrue, _, _ = skplot.ReadRSFFile(inittimetablepath, ndims = 3)
        TimetableSourceTrue = np.ascontiguousarray(np.transpose(TimetableSourceTrue, (2, 1, 0))) # nz, nx, nshot
    else:
        raise ValueError('inittimetablepath should be .npy or .rsf file')

    TimetableSourceTrue = TimetableSourceTrue[:, :, ::subsrcsample]
    TimetableSourceTrueDx = (TimetableSourceTrue[:, 2:, :] - TimetableSourceTrue[:, :-2, :])/(2*dx)
    TimetableSourceTrueDz = (TimetableSourceTrue[2:, :, :] - TimetableSourceTrue[:-2, :, :])/(2*dz)

    TimetableSourceTrueDx = np.pad(TimetableSourceTrueDx, ((0, 0), (1, 1), (0, 0)), 'edge')
    TimetableSourceTrueDz = np.pad(TimetableSourceTrueDz, ((1, 1), (0, 0), (0, 0)), 'edge')

    initveltensor = torch.from_numpy(initvel).float()
    inputoriveltensor = initveltensor.clone()[None, None, :, :]
    inputoriveltensor = inputoriveltensor.cuda(non_blocking = True)

    velmean = torch.mean(initveltensor).cuda(non_blocking = True) 
    vel_std = torch.std(initveltensor).cuda(non_blocking = True)
    initveltensor = ( initveltensor - torch.mean(initveltensor) ) / torch.std(initveltensor)

    initvelT = np.repeat(initvel[..., np.newaxis], srcxorr.size, axis=2)

    Z, X, SRX = np.meshgrid(z, x, srcxorr, indexing='ij')  # shapes: (nz, ncmp, nsrc)
    SRZ = np.ones(SRX.shape) * srz
    TimetableSourceprocess = TimetableSourceTrue.copy()

    T0, px0, pz0, TimetableSourceprocess = caculateT0_px_pz_sid(
        initvelT, SRX, SRZ, X, Z, zmin, xmin, dz, dx,
        TimetableSourceprocess, srcxorr, srz=srz)

    if stackprofilepath.endswith('.npy'):
        seismicprofile = np.load(stackprofilepath)
    elif stackprofilepath.endswith('.rsf'):
        seismicprofile, _, _ = skplot.ReadRSFFile(stackprofilepath, ndims = 2)
        seismicprofile = seismicprofile.T
    else:
        raise ValueError('stackprofilepath should be .npy or .rsf file')

    seismicprofile /= np.sqrt(np.mean(seismicprofile**2))
    skplot.visprofile(seismicprofile, os.path.join(exppath, 'ApplyFigure'), f'seismicprofile.png', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)', vmin = -2, vmax = 2)

    seismicprofile = torch.from_numpy(seismicprofile).float()
    tomo_model = networks.SCPINN(n1ForTime = 10, n1ForVel = 10, input_num_for_time = 3, input_num_for_vel = 2, scaleTime = 2, scaleVel = 2, outputfunc_time = torch.nn.Sigmoid(), outputfunc_vel = torch.nn.Tanh())

    tomo_model = tomo_model.to(device)

    if len(GPUIDS) > 1:
        logger.info('>>> use multi-GPU training ...')
        tomo_model = torch.nn.DataParallel(tomo_model, device_ids = GPUIDS, output_device = GPUIDS[0])

    current_state_dict = tomo_model.state_dict()

    DVGmodel_dict = torch.load(DVGModelPath) if os.path.isfile(DVGModelPath) else {}

    for key in DVGmodel_dict.keys():
        if len(GPUIDS) > 1:
            current_state_dict['module.netG.' + key] = DVGmodel_dict[key]
        else:
            current_state_dict['netG.' + key] = DVGmodel_dict[key]

    tomo_model.load_state_dict(current_state_dict)

    if config['continueTrain']:
        logger.info(f'>>> continue training, load model from {config["continueTrainModelPath"]} ...')
        tomo_model = skplot.load_networks(tomo_model, device, config['continueTrainModelPath'], continue_train = config['continueTrain'])

    print(f'seismic profile shape {seismicprofile.shape}\t initvel shape {initveltensor.shape}')
    INPUTIMGTEST = torch.cat((seismicprofile[None, :], initveltensor[None, :]), dim = 0)
    INPUTIMGTEST = INPUTIMGTEST[None,...].repeat(len(GPUIDS), 1, 1, 1)
    velmean = velmean[None, None, None, None].repeat(len(GPUIDS), 1, 1, 1)
    vel_std = vel_std[None, None, None, None].repeat(len(GPUIDS), 1, 1, 1)
    # inputGuidedTensor = inputGuidedTensor.repeat(len(GPUIDS), 1, 1, 1)
    INPUTIMGTEST = INPUTIMGTEST.cuda(non_blocking = True)
    seismicprofile = seismicprofile.cuda(non_blocking = True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, tomo_model.parameters()),
        lr=config['lr'],
        betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    if args.amp:
        logger.info('>>> Mixed precision AMP enabled.')
    rmseLoss = RMSELoss()

    # Step3: training
    totalTrainpoints = Z.size
    epochs = config["epochs"]

    bestTestLoss = np.inf
    valinteral = config['valinterval']

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-config['epoch_count'])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    schedulers = [lr_scheduler, warmup_scheduler]

    totalTestpoints = nz * ncmp  # retained for potential future validation usage

    crop_size = config["crop_size"]
    overlap = crop_size
    trainsrcsamplenum = config['trainsrcsamplenum']
    velLossceoff = config['velLossceoff']
    masklosscoeff = config['masklosscoeff']
    data_coeff = config['data_coeff']
    pde_coeff = config['pde_coeff']
    if config['DIPconstrain']:
        diplosscoeff = config['DIPconstraincoeff']
        tvlosscoeff = config['TVcoeff']

    savepath = os.path.join('result',Expname)
    writer = SummaryWriter(os.path.join(exppath,'tb_runs'))
    logger.info(f">>> start training ... {epochs} epochs in total\t ")
    total_parameters = sum(p.numel() for p in tomo_model.parameters())
    # logger total parameters in mb
    logger.info(f"total parameters: {total_parameters * 32 / 8 / 1024 / 1024} MB")

    # Pre-convert frequently accessed large numpy arrays to torch (device) to reduce per-iteration CPU->GPU transfers
    # Shapes: (nz, ncmp, nsrc)
    Z_torch = torch.from_numpy(Z).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    SRX_torch = torch.from_numpy(SRX).float().to(device)
    T0_torch = torch.from_numpy(T0).float().to(device)
    px0_torch = torch.from_numpy(px0).float().to(device)
    pz0_torch = torch.from_numpy(pz0).float().to(device)
    TimetableSourceprocess_torch = torch.from_numpy(TimetableSourceprocess).float().to(device)
    TimetableSourceTrueDx_torch = torch.from_numpy(TimetableSourceTrueDx).float().to(device)
    TimetableSourceTrueDz_torch = torch.from_numpy(TimetableSourceTrueDz).float().to(device)

    for epoch in range(config['epoch_count'], epochs):
        totalloss_Train = []
        total_data_loss_Train = []
        total_pde_loss_Train = []
        pde_time_src_loss_Train = []
        dz_loss_Train = []
        dx_loss_Train = []
        veloss_Train = []  # velocity consistency (label) loss collection
        if config['DIPconstrain']:
            Diploss_Train = []
            TVloss_Train = []
        optimizer.zero_grad()

        for start_zid in tqdm.trange(0, nz, overlap, desc='Epoch crop loop'):
            for start_xid in range(0, ncmp, overlap):
                endzid = min(start_zid + crop_size, nz)
                endxid = min(start_xid + crop_size, ncmp)
                init_crop = torch.squeeze(inputoriveltensor)[start_zid:endzid, start_xid:endxid]
                for srcid in range(0, srcxorr.size, trainsrcsamplenum):
                    endsrcid = min(srcid + trainsrcsamplenum, srcxorr.size)

                    # Slice & reshape (avoid new host->device copies)
                    zcorr = Z_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1).clone().requires_grad_()
                    xcorr = X_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1).clone().requires_grad_()
                    srcxcorrtain = SRX_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    t0train = T0_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    px0train = px0_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    pz0train = pz0_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    labelTimeTable = TimetableSourceprocess_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    timesourcedx = TimetableSourceTrueDx_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)
                    timesourcedz = TimetableSourceTrueDz_torch[start_zid:endzid, start_xid:endxid, srcid:endsrcid].reshape(-1, 1)

                    timesrcinput = torch.cat((zcorr, xcorr, srcxcorrtain), dim=1)
                    velinput = torch.cat((zcorr, xcorr), dim=1)

                    # Forward pass with autocast (model inference in mixed precision)
                    with torch.amp.autocast('cuda', enabled=args.amp):
                        time_src, nnoutputvel, dvgvelocity_all = tomo_model(
                            INPUTIMGTEST, velmean, vel_std, timesrcinput, velinput)

                    dvgvelocity_all = torch.mean(dvgvelocity_all, dim=0).squeeze()[start_zid:endzid, start_xid:endxid]
                    dvgvel = dvgvelocity_all[:, :, None].repeat(1, 1, endsrcid - srcid).reshape(-1, 1)
                    init_crop_rep = init_crop[:, :, None].repeat(1, 1, endsrcid - srcid).reshape(-1, 1)

                    vel_update = dvgvel + nnoutputvel
                    if config['DIPconstrain']:
                        velupdateReshape = vel_update.reshape(endzid - start_zid, endxid - start_xid, endsrcid - srcid).mean(dim=2)
                        seismicprofiletemp = seismicprofile[start_zid:endzid, start_xid:endxid]
                        diploss, tvloss = DipLossWithTV(seismicprofiletemp[None, None, ...], velupdateReshape[None, None, ...])
                        Diploss_Train.append(diploss.item())
                        TVloss_Train.append(tvloss.item())
                    else:
                        diploss = torch.tensor(0.0, device=device)
                        tvloss = torch.tensor(0.0, device=device)

                    # Compute losses; PDE/gradient part kept in full precision for stability
                    with torch.amp.autocast('cuda', enabled=False):
                        time_src_fp32 = time_src.float()
                        vel_update_fp32 = vel_update.float()
                        timedata_loss = rmseLoss(time_src_fp32, labelTimeTable)
                        loss_dict = DVGPDETomoGradientEnhancedLoss(
                            time_src_fp32, vel_update_fp32, zcorr.float(), xcorr.float(),
                            t0train.float(), px0train.float(), pz0train.float(), rmseLoss,
                            timesourcedx.float(), timesourcedz.float(),
                            vminvel=vminvel, vmaxvel=vmaxvel)
                    total_pde_loss = loss_dict['pde_residual_loss'] + loss_dict['dz_loss'] + loss_dict['dx_loss']

                    loss = (timedata_loss * data_coeff + total_pde_loss * pde_coeff +
                            diploss * (diplosscoeff if config['DIPconstrain'] else 0) +
                            tvloss * (tvlosscoeff if config['DIPconstrain'] else 0))

                    if not torch.isfinite(loss):
                        logger.error('Encountered NaN/Inf loss; aborting training to avoid corrupt states.')
                        raise FloatingPointError('loss is NaN/Inf')

                    if args.amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Metrics accumulation
                    totalloss_Train.append(loss.item())
                    total_data_loss_Train.append(timedata_loss.item())
                    total_pde_loss_Train.append(total_pde_loss.item())
                    pde_time_src_loss_Train.append(loss_dict['pde_residual_loss'].item())
                    dz_loss_Train.append(loss_dict['dz_loss'].item())
                    dx_loss_Train.append(loss_dict['dx_loss'].item())
                    veloss_Train.append(loss_dict['vel_consistency2'].item())

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        totalloss_Train = np.mean(totalloss_Train)
        total_data_loss_Train = np.mean(total_data_loss_Train)
        total_pde_loss_Train = np.mean(total_pde_loss_Train)
        pde_time_src_loss_Train = np.mean(pde_time_src_loss_Train)
        dz_loss_Train = np.mean(dz_loss_Train)
        dx_loss_Train = np.mean(dx_loss_Train)
        veloss_Train  = np.mean(veloss_Train) if len(veloss_Train) else -1

        writer.add_scalar('Total Loss', totalloss_Train, epoch)
        writer.add_scalar('Tomography Time Loss', total_data_loss_Train, epoch)
        writer.add_scalar('PDE Loss', total_pde_loss_Train, epoch)
        writer.add_scalar('PDE Time Loss', pde_time_src_loss_Train, epoch)
        writer.add_scalar('PDE dz Loss', dz_loss_Train, epoch)
        writer.add_scalar('PDE dx Loss', dx_loss_Train, epoch)
        writer.add_scalar('Velocity Loss', veloss_Train, epoch)

        if config['DIPconstrain']:
            Diploss_Train = np.mean(Diploss_Train)
            TVloss_Train = np.mean(TVloss_Train)
            writer.add_scalar('DIP Loss', Diploss_Train, epoch)
            writer.add_scalar('TV Loss', TVloss_Train, epoch)
        else:
            Diploss_Train = -1
            TVloss_Train = -1

        learning_rate = update_learningrate([optimizer], [schedulers], logger, lr_policy='warm_up')
        writer.add_scalar('Learning Rate', learning_rate, epoch)
        logger.info(
            f'>>>> epoch: {epoch}, train loss: {totalloss_Train:.4e}, data loss: {total_data_loss_Train:.4e}, '
            f'total pde loss: {total_pde_loss_Train:.4e}, pde time loss: {pde_time_src_loss_Train:.4e}, '
            f'dz loss: {dz_loss_Train:.4e}, dx loss: {dx_loss_Train:.4e}, vel loss: {veloss_Train:.4e}\t '
            f'lr = {learning_rate:.6e},\t dip loss: {Diploss_Train:.4e}, tv loss: {TVloss_Train:.4e}')

        if epoch % valinteral == 0 or epoch == epochs - 1 or epoch in savedepoch: # test
            logger.info('>>>> start testing ...')
            if not os.path.exists(os.path.join(exppath, 'model_saved')):
                os.makedirs(os.path.join(exppath, 'model_saved'), exist_ok = True)
                
            if epoch in savedepoch:
                if len(GPUIDS) > 1:
                    torch.save(tomo_model.module.state_dict(), os.path.join(exppath, 'model_saved', f'epoch{epoch}.pth'))
                else:
                    torch.save(tomo_model.state_dict(), os.path.join(exppath, 'model_saved', f'epoch{epoch}.pth'))

            nnvelocity_all = torch.zeros((nz, ncmp)).cuda(non_blocking = True)
            with torch.no_grad():
                if len(GPUIDS) > 1:
                    dvgvelocity_all = tomo_model.module.DVGforward(INPUTIMGTEST, velmean, vel_std).squeeze().detach()
                    dvgvelocity_all = torch.mean(dvgvelocity_all, dim = 0).squeeze().detach()
                else:
                    dvgvelocity_all = tomo_model.DVGforward(INPUTIMGTEST, velmean, vel_std).squeeze().detach()

                for start_zid in tqdm.trange(0, nz, crop_size):
                    for start_xid in range(0, ncmp, crop_size):
                        endzid = min(start_zid + crop_size, nz)
                        endxid = min(start_xid + crop_size, ncmp)
                        zcorr = ZGRID[start_zid:endzid, start_xid:endxid]
                        xcorr = XGRID[start_zid:endzid, start_xid:endxid]

                        zcorr = torch.from_numpy(zcorr.reshape(-1, 1)).float().cuda(non_blocking = True)
                        xcorr = torch.from_numpy(xcorr.reshape(-1, 1)).float().cuda(non_blocking = True)
                        velinput = torch.cat((zcorr, xcorr), dim = 1)
                        if len(GPUIDS) > 1:
                            nnvel = tomo_model.module.NNVelforward(velinput)
                        else:
                            nnvel = tomo_model.NNVelforward(velinput)
                        nnvel = nnvel.detach()
                        nnvel = nnvel.reshape((endzid - start_zid, endxid - start_xid))
                        nnvelocity_all[start_zid:endzid, start_xid:endxid] = nnvel

            velocity_all = dvgvelocity_all + nnvelocity_all

            velocity_all = velocity_all.cpu().numpy()
            nnvelocity_all = nnvelocity_all.cpu().numpy()
            dvgvelocity_all = dvgvelocity_all.cpu().numpy()
            
            if epoch in savedepoch:
                skplot.visVelocityprofile(velocity_all, os.path.join(exppath, 'ApplyFigure'), f'outvel_epoch{epoch}.jpg', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)')
                skplot.visVelocityprofile(velocity_all - initvel, os.path.join(exppath, 'ApplyFigure'), f'updateoutvel_epoch{epoch}.jpg', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)')
                skplot.visVelocityprofile(nnvelocity_all, os.path.join(exppath, 'ApplyFigure'), f'nnoutvel_epoch{epoch}.jpg', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)')
                skplot.visVelocityprofile(dvgvelocity_all, os.path.join(exppath, 'ApplyFigure'), f'DVGoutvel_epoch{epoch}.jpg', xmin, dx, zmin, dz, figsize = skplot.cm2inch(20, 7.5), fontsize = 10, labelsize = 10, xlabelname = 'Lateral (km)', ylabelname = 'Depth (km)')
            
            if truevel is not None:
                trueloss.append(np.sqrt(np.mean((velocity_all - truevel)**2)))
                trueepoch.append(epoch)
                writer.add_scalar('True Velocity Loss', trueloss[-1], epoch)
            if epoch in savedepoch:
                if not os.path.exists(os.path.join(exppath, 'Inference')):
                    os.makedirs(os.path.join(exppath, 'Inference'), exist_ok = True)
                np.save(os.path.join(exppath, 'Inference', f'epoch{epoch}_outvel.npy'), velocity_all)
                np.save(os.path.join(exppath, 'Inference', f'epoch{epoch}_nnoutvel.npy'), nnvelocity_all)
                np.save(os.path.join(exppath, 'Inference', f'epoch{epoch}_dvgoutvel.npy'), dvgvelocity_all)

    writer.close()
    logger.info('>>>> Finish training ...')
    np.save(os.path.join(exppath, f'epoch_outvel.npy'), velocity_all)
    np.save(os.path.join(exppath, f'epoch_nnoutvel.npy'), nnvelocity_all)
    np.save(os.path.join(exppath, f'epoch_dvgoutvel.npy'), dvgvelocity_all)

    if truevel is not None:
        figure, ax = plt.subplots(1, 1, figsize = skplot.cm2inch(10, 7.5))
        ax.plot(trueepoch, trueloss)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('True velocity MSE Loss')
        figure.savefig(os.path.join(exppath, 'ApplyFigure', 'TruevelMSELoss.png'))
