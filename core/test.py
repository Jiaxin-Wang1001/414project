import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.decoder2 import Decoder2
from utils.average_meter import AverageMeter
import os
from tensorboardX import SummaryWriter

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKER,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)


        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()


        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()

    # test_writer = SummaryWriter("/content/drive/Shareddrives/CMPUT_414_1/414project_1/outputsforreport/two_headed_out_put")
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, projections_images) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)
            projections_images = utils.helpers.var_or_cuda(projections_images)
            # Test the encoder, decoder
            image_features = encoder(rendering_images)
            generated_volume, generated_projections = decoder(image_features)

            generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss1 = bce_loss(generated_volume, ground_truth_volume) * 10 
            # print("+++++++++++++++++++++++++++++++++++++++")
            # print(generated_projections.shape)
            # print(projections_images.shape)
            # print("---------------------------------------")
            encoder_loss2 = bce_loss(generated_projections, projections_images) * 10
            # encoder_loss = encoder_loss1 + encoder_loss2
            encoder_loss = encoder_loss1 + encoder_loss2
            # if encoder_loss1.item() > encoder_loss2.item():
            #     encoder_loss = encoder_loss1
            # else:
            #     encoder_loss = encoder_loss2


            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            

            # Append generated volumes to TensorBoard
            if test_writer and sample_idx < 3:
                # Volume Visualization
                rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
                # print("lalala", rendering_views.shape)
                test_writer.add_images('Model%02d/Input' % sample_idx, torch.squeeze(rendering_images), epoch_idx, dataformats="CWH")
                test_writer.add_images('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx, dataformats="HWC")
                rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
                # print("lalala2", rendering_views.shape)
                test_writer.add_images('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx, dataformats="HWC")

            # Print sample loss and IoU
            # logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
            #              (sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
            #               refiner_loss.item(), ['%.4f' % si for si in sample_iou]))
            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s loss1 = %.4f loss2 = %.4f EDLoss = %.4f IoU = %s' %
                        (sample_idx + 1, n_samples, taxonomy_id, sample_name,  encoder_loss1.item(), encoder_loss2.item(), encoder_loss.item(),
                         ['%.4f' % si for si in sample_iou]))
    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Decoder/IoU', max_iou, epoch_idx)

    return max_iou
