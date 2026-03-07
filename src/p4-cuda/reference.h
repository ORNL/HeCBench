#include <algorithm>
#include <cstdlib>

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

        bndbox_output[0] = resCount+1;
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

  typedef struct {
    float val[9];
  } combined_float;

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

  bool ok = bndbox_num_r == bndbox_num;
  if (ok) {
    // note the max score is not necessarily unique
    auto max_ptr = std::max_element(score_output, score_output + bndbox_num);
    int loc = max_ptr - score_output;
    auto cbndbox = (combined_float*)bndbox_output + loc;
    
    for (int i = 0; i < bndbox_num_r; i++) {
      if (fabsf(*max_ptr - score_output_r[i]) < 1e-6f) {
        auto cbndbox_r = (combined_float*)bndbox_output_r + i;
        if (cbndbox->val[8] == cbndbox_r->val[8]) { // find a max score that matches 
          printf("Comparing values at location %d and reference location %d\n", loc, i);
          for (int i = 0; i < 9; i++) {
            if (fabsf(cbndbox->val[i] - cbndbox_r->val[i]) > 1e-3f) {
              ok = false;
              break;
            }
          }
          break;
        }
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  free(score_output_r);
  free(bndbox_output_r);
}
