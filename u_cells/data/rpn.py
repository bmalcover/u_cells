# -*- coding: utf-8 -*-
""" Module containing all the RPN data classes.

The RPN data classes are used to store the data of bounding boxes and its related masks, its based
on the Mask R-CNN implementation of the original paper.

Writen by: Miquel Miró Nicolau (UIB)
"""
from abc import ABC, abstractmethod
from typing import Tuple
import warnings

import numpy as np
import math
import skimage
import imgaug

from tensorflow.keras import utils as KU

from ..common import utils


class Dataset(ABC):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset you want to use.
    For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    @abstractmethod
    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with a binary mask per
                   instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        warnings.warn(
            "You are using the default load_mask(), maybe you need to define your own one.",
            RuntimeWarning)

        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)

        return mask, class_ids


class DataGenerator(KU.Sequence):
    """ An iterable that returns images and corresponding target class ids,
    bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
    when multiprocessing=True.

    Args:
        dataset: The Dataset object to pick data from
        config: The model config object
        shuffle: If True, shuffles the samples before every epoch
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation. For
            example, passing imgaug.augmenters.Fliplr(0.5) flips images right/left 50% of the time.
        detection_targets: If True, generate detection targets (class IDs, bbox deltas, and masks).
            Typically for debugging or visualizations because in trainig detection targets are
            generated by DetectionTargetLayer.

    Returns:
        Python iterable. Upon calling __getitem__() on it, the iterable returns two lists, inputs
        and outputs. The contents of the lists differ depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets is True then
                the outputs list contains target class_ids, bbox deltas, and masks.
    """

    def __init__(self, steps: int, dataset, config, shuffle=True, augmentation=None,
                 detection_targets=False):

        self.__steps = steps
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.backbone_shapes = DataGenerator.__compute_backbone_shapes(config.IMAGE_SHAPE,
                                                                       config.BACKBONE_STRIDES)
        self.anchors = DataGenerator.__generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                config.RPN_ANCHOR_RATIOS,
                                                                self.backbone_shapes,
                                                                config.BACKBONE_STRIDES,
                                                                config.RPN_ANCHOR_STRIDE)

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.batch_size = self.config.BATCH_SIZE
        self.detection_targets = detection_targets

    def __len__(self):
        return self.__steps

    def __getitem__(self, idx):
        b = 0
        image_index = -1
        while b < self.batch_size:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(self.image_ids)

            if self.shuffle and image_index == 0:
                np.random.shuffle(self.image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = self.image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = DataGenerator.load_image_gt(
                self.dataset, self.config, image_id, augmentation=self.augmentation)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = DataGenerator.build_rpn_targets(image.shape, self.anchors,
                                                                  gt_class_ids, gt_boxes,
                                                                  self.config)

            # Init batch arrays
            if b == 0:
                batch_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4],
                    dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (self.batch_size,) + image.shape, dtype=np.float32)
                if self.config.COMBINE_FG:
                    mask_depth = 1
                else:
                    mask_depth = self.config.MAX_GT_INSTANCES
                batch_gt_masks = np.zeros(
                    (self.batch_size, gt_masks.shape[0], gt_masks.shape[1], mask_depth),
                    dtype=gt_masks.dtype)
                batch_gt_class_ids = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)

                # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = self.mold_image(image)
            gt_masks = gt_masks.reshape((gt_masks.shape[0], gt_masks.shape[1], -1))
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            b += 1

        if self.config.DO_MASK:
            inputs = [batch_images, batch_gt_masks, batch_rpn_match, batch_rpn_bbox,
                      batch_gt_class_ids]
        else:
            inputs = [batch_images, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids]

        outputs = [np.zeros((4, 512, 512, 100))] + (
                    [np.zeros((10, 10))] * (self.config.RPN_NUM_OUTPUTS - 1))

        return inputs, outputs

    @staticmethod
    def __generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
        """ Generates the anchors on each pixel of the feature map.

        Args:
            scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
            ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
            shape: [height, width] spatial shape of the feature map over which to generate anchors.
            feature_stride: Stride of the feature map relative to the image in pixels.
            anchor_stride: Stride of anchors on the feature map. For example, if the value is 2 then
                            generate anchors for every other feature map pixel.

        Returns:

        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    @staticmethod
    def __generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
        """ Generate anchors at different levels of a feature pyramid.

        Each scale is associated with a level of the pyramid, but each ratio is used in all levels
        of the pyramid.

        Returns:
            anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted with the same
                    order of the given scales. So, anchors of scale[0] come first, then anchors of
                    scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(DataGenerator.__generate_anchors(scales[i], ratios, feature_shapes[i],
                                                            feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    @staticmethod
    def __compute_backbone_shapes(image_shape, strides):
        """ Computes the width and height of each stage of the backbone network.

        Args:
            image_shape: [height, width, depth].
            strides: Strides for each feature map.

        Returns:
            [N, (height, width)]. Where N is the number of stages
        """
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))] for
             stride in strides])

    @staticmethod
    def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
        """ Builds the targets for the Region Proposal Network.

        Given the anchors and GT boxes, compute overlaps and identify positive anchors and deltas to
        refine them to match their corresponding GT boxes.

        Args:
            image_shape: [height, width, depth]
            anchors: [num_anchors, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes] Integer class IDs.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
            rpn_match: [N] (int32) matches between anchors and GT boxes.
                       1 = positive anchor, -1 = negative anchor, 0 = neutral
            rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        """
        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
        # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
        rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = np.where(gt_class_ids < 0)[0]
        if crowd_ix.shape[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = np.where(gt_class_ids > 0)[0]
            crowd_boxes = gt_boxes[crowd_ix]
            gt_class_ids = gt_class_ids[non_crowd_ix]
            gt_boxes = gt_boxes[non_crowd_ix]
            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
            crowd_iou_max = np.amax(crowd_overlaps, axis=1)
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = utils.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        #
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in crowd areas.
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU match all of them
        gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
        rpn_match[gt_iou_argmax] = 1
        # 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= 0.7] = 1

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = np.where(rpn_match == 1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0
        # Same for negative proposals
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                            np.sum(rpn_match == 1))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = np.where(rpn_match == 1)[0]
        ix = 0  # index into rpn_bbox
        # TODO: use box_refinement() rather than duplicating the code here
        for i, a in zip(ids, anchors[ids]):
            # Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[i]]

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            rpn_bbox[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]
            # Normalize
            rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
            ix += 1

        return rpn_match, rpn_bbox

    @staticmethod
    def load_image_gt(dataset, config, image_id, augmentation=None):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        Args:
            dataset:
            config:
            image_id:
            augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                          For example, passing imgaug.augmenters.Fliplr(0.5) flips images right/left
                          50% of the time.

        Returns:
            image: [height, width, 3]
            shape: the original shape of the image before resizing and cropping.
            class_ids: [instance_count] Integer class IDs
            bbox: [instance_count, (y1, x1, y2, x2)]
            mask: [height, width, instance_count]. The height and width are those of the image
                  unless use_mini_mask is True, in which case they are defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        original_shape = image.shape
        image, window, scale, padding, crop = utils.resize_image(image,
                                                                 min_dim=config.IMAGE_MIN_DIM,
                                                                 min_scale=config.IMAGE_MIN_SCALE,
                                                                 max_dim=config.IMAGE_MAX_DIM,
                                                                 mode=config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding, crop)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask,
                                     hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            mask = mask

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]

        if config.RANDOM_MASKS:
            order_mask = np.arange(mask.shape[-1])
            np.random.shuffle(order_mask)

            mask = mask[:, :, order_mask]

        bbox = utils.extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
        source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        # Image meta data
        image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                        window, scale, active_class_ids)

        if config.COMBINE_FG or config.MAKE_BACKGROUND_MASK:
            foreground_mask = np.sum(mask, axis=-1)
            foreground_mask[foreground_mask > 1] = 1

            if config.COMBINE_FG:
                mask = foreground_mask

            if config.MAKE_BACKGROUND_MASK:
                mask = np.concatenate([1 - foreground_mask, mask], axis=-1)

        return image, image_meta, class_ids, bbox, mask

    def mold_image(self, images):
        """ Mold inputs to format expected by the neural network.

        Normalization is done based on the mean and std of the image dataset. This mean and std are
        both stored in the config object and precalculated.

        Args:
            images: List of image matrices [height, width, channels]. Expects an RGB (in that
                    specific order (or array of images) and subtracts the mean pixel and converts
                    it to float.
        """
        return images.astype(np.float64) - self.config.MEAN_PIXEL

    def decode_deltas(self, deltas: np.ndarray):
        """ Decodes deltas prediction to the bounding boxes.

        The deltas represent the changes to be applied to the anchors to obtain a better bounding
        box. This delta [dy, dx, dh, dw] are:
            dy = (gt_y - a_y) / a_h
            dx = (gt_x - a_x) / a_w
            dh = ln(gt_h / a_h)
            dw = ln(gt_w / a_w)

        where "a" is the anchor previously defined, "d" are the deltas passed as parameter and "gt"
        are the refined bounding boxes. This function aims from "d" and "a" obtain "gt". For this
        reason what it does is reverse the operations previously defined.

        Args:
            deltas (np.array): [dy, dx, dh, dw]

        Returns:

        """
        assert deltas.shape == self.anchors.shape, "Deltas and anchors has different size"

        b_boxes = np.zeros_like(deltas)
        deltas = np.copy(deltas)

        # From log(w) and log(h) to w and h

        deltas[:, 2] = np.e ** deltas[:, 2]
        deltas[:, 3] = np.e ** deltas[:, 3]

        # Anchors shape
        anchor_height = self.anchors[:, 2] - self.anchors[:, 0]
        anchor_width = self.anchors[:, 3] - self.anchors[:, 1]

        # Refined shape
        height = anchor_height * deltas[:, 2]  # e^0 = 1
        width = anchor_width * deltas[:, 3]  # e^0 = 1

        center_y = (deltas[:, 0] * anchor_height) + (self.anchors[:, 0] + 0.5 * anchor_height)
        center_x = (deltas[:, 1] * anchor_width) + (self.anchors[:, 1] + 0.5 * anchor_width)

        b_boxes[:, 0] = center_y - 0.5 * height
        b_boxes[:, 1] = center_x - 0.5 * width
        b_boxes[:, 2] = b_boxes[:, 0] + height
        b_boxes[:, 3] = b_boxes[:, 1] + width

        return b_boxes


def compose_image_meta(image_id: int, original_image_shape: Tuple[int, int, int],
                       image_shape: Tuple[int, int, int], window: Tuple[int, int, int, int], scale,
                       active_class_ids):
    """ Takes attributes of an image and puts them in one 1D array.

    Args:
        image_id: An int ID of the image. Useful for debugging.
        original_image_shape: [H, W, C] before resizing or padding.
        image_shape: [H, W, C] after resizing and padding
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
        scale: The scaling factor applied to the original image (float32)
        active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.

    Returns:

    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta
