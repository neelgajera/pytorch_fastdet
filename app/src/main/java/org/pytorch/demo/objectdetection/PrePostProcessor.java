// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import static java.lang.Math.pow;


import android.graphics.Rect;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }
    String get_string(){
        return rect.toString() + score.toString() ;
    }

}

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    static int mInputWidth = 352;
    static int mInputHeight = 352;

    // model output is of size 25200*(num_of_class+5)
    private static int mOutputRow = 22; // as decided by the YOLOv5 model for input image of size 640*640
//    private static int mOutputColumn = 6; // left, top, right, bottom, score and 80 class probability
    private static float mThreshold = 0.80f; // score above which a detection is generated
    private static int mNmsLimit = 15;

    static String[] mClasses;

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<Result> nonMaxSuppression(ArrayList<Result> boxes, int limit, float threshold) {

        // Do an argsort on the confidence scores, from high to low.
        Collections.sort(boxes,
                new Comparator<Result>() {
                    @Override
                    public int compare(Result o1, Result o2) {
                        return o1.score.compareTo(o2.score);
                    }
                });

        ArrayList<Result> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i=0; i<boxes.size() && !done; i++) {
            if (active[i]) {
                Result boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<boxes.size(); j++) {
                    if (active[j]) {
                        Result boxB = boxes.get(j);
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }
    static float Sigmoid(float x)
    {
        return (float) (1.0f / (1.0f + Math.exp(-x)));
    }
    static float Tenh(float x)
    {
        return (float) (2.0f / (1.0f + Math.exp(-2*x)) -1);
    }


    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<Result> outputsToNMSPredictions(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {
        ArrayList<Result> results = new ArrayList<>();

        for (int h = 0;h< mOutputRow; h++) {
        for (int w = 0; w< mOutputRow; w++) {
            int obj_score_index = (0 * mOutputRow * mOutputRow) + (h * mOutputRow) + w;
            float obj_score = outputs[obj_score_index];
            int cls = 0;
//            float max_score = 0.0f;
//            for multiclass detection
//            for (size_t i = 0; i < class_num; i++)
//            {
//                int obj_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w;
//                float cls_score = output[obj_score_index];
//                if (cls_score > max_score)
//                {
//                    max_score = cls_score;
//                    cls = i;
//                }
//            }
            obj_score_index = (5* mOutputRow * mOutputRow) + (h * mOutputRow) + w;
            float cls_score = outputs[obj_score_index];
            double score = pow(cls_score, 0.4) * pow(obj_score, 0.6);
            float floatscore = (float)score;


            if (floatscore > mThreshold) {

                int x_offset_index = (1 * mOutputRow * mOutputRow) + (h * mOutputRow) + w;
                int y_offset_index = (2 * mOutputRow * mOutputRow) + (h * mOutputRow) + w;
                int box_width_index = (3 * mOutputRow * mOutputRow) + (h * mOutputRow) + w;
                int box_height_index = (4 * mOutputRow * mOutputRow) + (h * mOutputRow) + w;

                float x_offset =  Tenh(outputs[x_offset_index]);
                float y_offset =  Tenh(outputs[y_offset_index]);
                float box_width = Sigmoid(outputs[box_width_index]);
                float box_height = Sigmoid(outputs[box_height_index]);

                float cx = (w + x_offset) / mOutputRow;
                float cy = (h + y_offset) / mOutputRow;

                float left = (float) (imgScaleX * (cx - box_width * 0.5));
                float top = (float) (imgScaleY * (cy - box_height * 0.5));
                float right = (float) (imgScaleX * (cx + box_width * 0.5));
                float bottom = (float) (imgScaleY * (cy + box_height * 0.5));

                Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
                Result result = new Result(cls, floatscore, rect);
                results.add(result);
            }
        }}
        return nonMaxSuppression(results, mNmsLimit, mThreshold);
    }


}
