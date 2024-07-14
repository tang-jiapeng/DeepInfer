#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/tensor.h"

TEST(test_flatten, flatten1) {
  using namespace deep_infer;
  Tensor<float> f1(2, 3, 4);
  LOG(INFO) << "-------------------before Flatten-------------------";
  f1.Show();
  f1.Flatten(true);
  LOG(INFO) << "-------------------after Flatten-------------------";
  f1.Show();
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24);
}

TEST(test_flatten, flatten2) {
  using namespace deep_infer;
  Tensor<float> f1(12, 24);
  LOG(INFO) << "-------------------before Flatten-------------------";
  f1.Show();
  f1.Flatten(true);
  LOG(INFO) << "-------------------after Flatten-------------------";
  f1.Show();
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
}

TEST(test_padding, padding1) {
  using namespace deep_infer;
  Tensor<float> tensor(3, 4, 5);
  // channels,rows,cols
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  LOG(INFO) << "-------------------before padding-------------------";
  tensor.Show();
  tensor.Padding({1, 2, 3, 4}, 0);
  LOG(INFO) << "-------------------after padding-------------------";
  tensor.Show();
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if ((r >= 1 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                              << " " << r << " " << c_;
        }
        index += 1;
      }
    }
  }
}

TEST(test_padding, padding2) {
  using namespace deep_infer;
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor.rows(), 8);
  ASSERT_EQ(tensor.cols(), 9);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}