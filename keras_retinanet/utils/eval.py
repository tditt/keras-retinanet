"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import numpy as np
import os

import cv2


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.5, max_detections=500, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        #filter out boxes that are too small
        box_minimum = 18
        valid_box_indices = []
        for b, box in enumerate(boxes[0]):
            width = box[2] - box[0]
            height = box[3] - box[1]
            if width > box_minimum and height > box_minimum:
                valid_box_indices.append(b)

        valid_boxes = boxes[0][valid_box_indices]
        valid_scores = scores[0][valid_box_indices]
        valid_labels = labels[0][valid_box_indices]

        # correct boxes for image scale
        valid_boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(valid_scores > score_threshold)[0]

        # select those scores
        valid_scores = valid_scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-valid_scores)[:max_detections]

        # select detections
        image_boxes = valid_boxes[indices[scores_sort]]
        image_scores = valid_scores[scores_sort]
        image_labels = valid_labels[indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations


def _eval_iou(iou, generator, all_detections, all_annotations):
    metrics = {}
    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        num_detections = 0.0
        # loop all images
        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            num_detections += detections.shape[0]
            detected_annotations = []
            # loop all detections in image
            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            metrics[label] = 0
            continue
        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        # calculate percentage of "of xx detections, xx are true positives"
        true_pos_percentage = np.mean(true_positives)
        # calculate percentage of "of xx detections, xx are false positives"
        false_pos_percentage = np.mean(false_positives)
        true_pos_count = np.sum(true_positives)
        false_pos_count = np.sum(false_positives)
        false_neg_count = (num_annotations - np.sum(true_positives))
        false_neg_percentage = false_neg_count / num_annotations
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall precision f1
        precision = true_pos_count / (true_pos_count + false_pos_count)
        recall = true_pos_count / (true_pos_count + false_neg_count)
        if recall == 0 or precision == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        recall_arr = true_positives / num_annotations
        precision_arr = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall_arr, precision_arr)
        metrics[label] = (iou,
                          num_annotations, num_detections, true_pos_count, true_pos_percentage, false_pos_count,
                          false_pos_percentage,
                          false_neg_count, false_neg_percentage, precision, recall, f1, average_precision
                          )
    return metrics


def _print_to_log(*args, **kwargs):
    print(*args, **kwargs)
    with open('evaluation.log', 'a') as file:
        print(*args, **kwargs, file=file)


def _print_evaluation(metrics, generator):
    present_classes = 0
    all_classes_ap = 0
    all_classes_ap50 = 0
    all_classes_ap75 = 0
    _print_to_log('############################## Evaluation Results #################################')
    for label, (eval_metrics, ap, ap50, ap75) in metrics.items():
        (iou, num_annotations, num_detections, true_pos_count, true_pos_percentage, false_pos_count,
         false_pos_percentage, false_neg_count, false_neg_percentage, precision, recall, f1,
         average_precision) = eval_metrics
        _print_to_log('---------------------------------class ', generator.label_to_name(label))
        _print_to_log('{:.0f} annotations of class'.format(num_annotations), generator.label_to_name(label), 'with:')
        _print_to_log('{:.0f} annotations not detected'.format(false_neg_count),
                      '({:.4f} false negatives)'.format(false_neg_percentage))
        _print_to_log('{:.0f} detections of class'.format(num_detections), generator.label_to_name(label), 'with:')
        _print_to_log('IOU of {:.2f}'.format(iou))
        _print_to_log('{:.0f} matches'.format(true_pos_count), '({:.4f} true positives)'.format(true_pos_percentage))
        _print_to_log('{:.0f} false detections'.format(false_pos_count),
                      '({:.4f} false positives)'.format(false_pos_percentage))
        _print_to_log('precision: {:.4f}'.format(precision))
        _print_to_log('recall: {:.4f}'.format(recall))
        _print_to_log('f1 score: {:.4f}'.format(f1))
        _print_to_log('average precision for this IOU: {:.4f}'.format(average_precision))
        _print_to_log('class mAP: {:.4f}'.format(ap), 'class mAP50: {:.4f}'.format(ap50),
                      'class mAP75: {:.4f}'.format(ap75))
        _print_to_log('-----------------------------------------------------------------------')
        if num_annotations > 0:
            present_classes += 1
            all_classes_ap += ap
            all_classes_ap50 += ap50
            all_classes_ap75 += ap75
    _print_to_log('all classes: mAP : {:.4f}'.format(all_classes_ap / present_classes),
                  'mAP50 : {:.4f}'.format(all_classes_ap50 / present_classes),
                  'mAP75 : {:.4f}'.format(all_classes_ap75 / present_classes))
    _print_to_log('#################################################################################')


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=500,
        save_path=None,
        print_evaluation=True
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path)
    all_annotations = _get_annotations(generator)

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    eval_metrics = {}
    average_precisions = np.zeros((10, generator.num_classes()))
    count = 0
    for i in range(50, 100, 5):
        iou = i / 100
        iou_metrics = _eval_iou(iou, generator, all_detections, all_annotations)
        for label in range(generator.num_classes()):
            average_precisions[count][label] = iou_metrics[label][-1]
        if iou == iou_threshold:
            eval_metrics = iou_metrics
        count += 1
    all_metrics = {}
    ap = np.mean(average_precisions, axis=0)
    ap50 = average_precisions[0]
    ap75 = average_precisions[5]

    for label in range(generator.num_classes()):
        all_metrics[label] = (eval_metrics[label], ap[label], ap50[label], ap75[label])

    if print_evaluation:
        _print_evaluation(all_metrics, generator)

    return all_metrics
