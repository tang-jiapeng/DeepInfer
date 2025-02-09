#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/tensor.h"

float MinusOne(float value) { return value - 1.f; }

TEST(test_transform, transform1) {
  using namespace deep_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  f1.Transform(MinusOne);
  f1.Show();
}
