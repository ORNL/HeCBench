#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int reference (
  const float *__restrict__ cls_input,
        float *__restrict__ box_input,
  const float *__restrict__ dir_cls_input,
  const float *__restrict__ anchors,
  const float *__restrict__ anchor_bottom_heights,
        float *__restrict__ bndbox_output,
        float *__restrict__ score_output,
  const float min_x_range,
  const float max_x_range,
  const float min_y_range,
  const float max_y_range,
  const int feature_size,
  const int feature_x_size,
  const int feature_y_size,
  const int num_anchors,
  const int num_classes,
  const int num_box_values,
  const float score_thresh,
  const float dir_offset)
{
  int resCount = 0;
  for (int loc_index = 0; loc_index < feature_size; loc_index++) {
    int col = loc_index % feature_x_size;
    int row = loc_index / feature_x_size;
    float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
    float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);

    for (int itanchor = 0; itanchor < num_anchors; itanchor++) {
      int cls_offset = loc_index * num_anchors * num_classes + itanchor * num_classes;
      float dev_cls[2] = {-1.f, 0.f};

      const float *scores = cls_input + cls_offset;
      float max_score = 1.f / (1.f + expf(-scores[0]));
      int cls_id = 0;
      for (int i = 1; i < num_classes; i++) {
        float cls_score = 1.f / (1.f + expf(-scores[i]));
        if (cls_score > max_score) {
          max_score = cls_score;
          cls_id = i;
        }
      }
      dev_cls[0] = static_cast<float>(cls_id);
      dev_cls[1] = max_score;

      if (dev_cls[1] >= score_thresh)
      {
        const int box_offset = loc_index * num_anchors * num_box_values + itanchor * num_box_values;
        const int dir_cls_offset = loc_index * num_anchors * 2 + itanchor * 2;
        const float *anchor_ptr = anchors + itanchor * 4;
        const float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[itanchor / 2];
        const float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
        float *box_encodings = box_input + box_offset;

        const float xa = anchor[0];
        const float ya = anchor[1];
        const float za = anchor[2];
        const float dxa = anchor[3];
        const float dya = anchor[4];
        const float dza = anchor[5];
        const float ra = anchor[6];
        const float diagonal = sqrtf(dxa * dxa + dya * dya);
        box_encodings[0] = box_encodings[0] * diagonal + xa;
        box_encodings[1] = box_encodings[1] * diagonal + ya;
        box_encodings[2] = box_encodings[2] * dza + za;
        box_encodings[3] = expf(box_encodings[3]) * dxa;
        box_encodings[4] = expf(box_encodings[4]) * dya;
        box_encodings[5] = expf(box_encodings[5]) * dza;
        box_encodings[6] = box_encodings[6] + ra;

        const int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
        const float period = (float)M_PI;
        const float val = box_input[box_offset + 6] - dir_offset;
        const float dir_rot = val - floorf(val / (period + 1e-8f)) * period;
        const float yaw = dir_rot + dir_offset + period * dir_label;

        float *data = bndbox_output + resCount * 9;
        data[0] = box_input[box_offset];
        data[1] = box_input[box_offset + 1];
        data[2] = box_input[box_offset + 2];
        data[3] = box_input[box_offset + 3];
        data[4] = box_input[box_offset + 4];
        data[5] = box_input[box_offset + 5];
        data[6] = yaw;
        data[7] = dev_cls[0];
        data[8] = box_offset;
        score_output[resCount++] = max_score;
      }
    }
  }
  return resCount;
}

void verify (
  const int bndbox_num,
  const float *__restrict__ cls_input,
        float *__restrict__ box_input,
  const float *__restrict__ dir_cls_input,
  const float *__restrict__ anchors,
  const float *__restrict__ anchor_bottom_heights,
        float *__restrict__ bndbox_output,
        float *__restrict__ score_output,
  const float min_x_range,
  const float max_x_range,
  const float min_y_range,
  const float max_y_range,
  const int feature_size,
  const int feature_x_size,
  const int feature_y_size,
  const int num_anchors,
  const int num_classes,
  const int num_box_values,
  const float score_thresh,
  const float dir_offset)
{
  const int feature_anchor_size = feature_size * num_anchors;
  const int score_size_byte = feature_anchor_size * sizeof(float);

  const int bndbox_size = feature_anchor_size * 9;
  const int bndbox_size_byte = bndbox_size * sizeof(float);

  float *score_output_r = (float*) malloc (score_size_byte);
  float *bndbox_output_r = (float*) malloc (bndbox_size_byte);

  int bndbox_num_r = reference(
      cls_input,
      box_input,
      dir_cls_input,
      anchors,
      anchor_bottom_heights,
      bndbox_output_r,
      score_output_r,
      min_x_range,
      max_x_range,
      min_y_range,
      max_y_range,
      feature_size,
      feature_x_size,
      feature_y_size,
      num_anchors,
      num_classes,
      num_box_values,
      score_thresh,
      dir_offset);

  bool ok = true;
  if (bndbox_num != bndbox_num_r) {
    printf("Box count mismatch: device %d reference %d\n", bndbox_num, bndbox_num_r);
    ok = false;
  } else if (bndbox_num > 0) {
    // atomicAdd stores boxes in an arbitrary order. Match on the box offset,
    // which is a unique anchor index, and compare every box.
    std::vector<int> dev_order(bndbox_num), ref_order(bndbox_num);
    for (int i = 0; i < bndbox_num; i++) {
      dev_order[i] = i;
      ref_order[i] = i;
    }
    auto offset_less = [](const float* boxes, int a, int b) {
      return boxes[a * 9 + 8] < boxes[b * 9 + 8];
    };
    std::sort(dev_order.begin(), dev_order.end(),
              [&](int a, int b) { return offset_less(bndbox_output, a, b); });
    std::sort(ref_order.begin(), ref_order.end(),
              [&](int a, int b) { return offset_less(bndbox_output_r, a, b); });

    for (int n = 0; n < bndbox_num; n++) {
      const int d = dev_order[n];
      const int r = ref_order[n];
      const float dev_offset = bndbox_output[d * 9 + 8];
      const float ref_offset = bndbox_output_r[r * 9 + 8];
      if (dev_offset != ref_offset) {
        printf("Box offset mismatch at sorted index %d: device %f reference %f\n",
               n, dev_offset, ref_offset);
        ok = false;
        break;
      }
      if (n + 1 < bndbox_num &&
          bndbox_output[dev_order[n + 1] * 9 + 8] == dev_offset) {
        printf("Duplicate box offset %f\n", dev_offset);
        ok = false;
        break;
      }
      for (int k = 0; k < 7; k++) {
        if (fabsf(bndbox_output[d * 9 + k] - bndbox_output_r[r * 9 + k]) > 1e-3f) {
          printf("Box %f component %d mismatch: %f %f\n",
                 dev_offset, k,
                 bndbox_output[d * 9 + k], bndbox_output_r[r * 9 + k]);
          ok = false;
          break;
        }
      }
      if (!ok) break;
      if (bndbox_output[d * 9 + 7] != bndbox_output_r[r * 9 + 7]) {
        printf("Box %f class mismatch: %f %f\n",
               dev_offset, bndbox_output[d * 9 + 7], bndbox_output_r[r * 9 + 7]);
        ok = false;
        break;
      }
      if (fabsf(score_output[d] - score_output_r[r]) > 1e-4f) {
        printf("Box %f score mismatch: %f %f\n",
               dev_offset, score_output[d], score_output_r[r]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  free(score_output_r);
  free(bndbox_output_r);
}
