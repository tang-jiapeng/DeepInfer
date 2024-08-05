#pragma once

#include <armadillo>
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>

namespace deep_infer {

class CSVDataLoader {
 public:
  /**
   * 从csv文件中初始化张量
   * @param file_path csv文件的路径
   * @param split_char 分隔符号
   * @return 根据csv文件得到的张量
   */
  static arma::fmat LoadData(const std::string& file_path, char split_char = ',');

 private:
  /**
   * 得到csv文件的尺寸大小，LoadData中根据这里返回的尺寸大小初始化返回的fmat
   * @param file csv文件的路径
   * @param split_char 分割符号
   * @return 根据csv文件的尺寸大小
   */
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream& file, char split_char);
};

}  // namespace deep_infer
