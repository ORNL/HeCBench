#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include "hip_error.hpp"
#include "general.hpp"
#include "kernel.hpp"

namespace SNICIT_BEY{

class SNICIT{

  private:
    std::vector<float*> _dev_Y_hidden; // 2 buffers
    std::vector<int*> _dev_hidden_roffw; // hidden layer W row offset
    std::vector<int*> _dev_hidden_colsw; // hidden layer W cols
    std::vector<float*> _dev_hidden_valsw; // hidden layer W vals
    std::vector<float*> _dev_hidden_bias; // hidden layer W vals


    float *Y_input; // on cpu
    float *_dev_Y_output_whole; // on gpu

    float* _dev_Y_input;
    float* _dev_Y_output;
    float* _dev_input_weight;
    float* _dev_input_bias;
    float* _dev_output_weight;
    float* _dev_output_bias;
    int* _dev_result_label;
    


    std::string weight_path, bias_path;
    int num_hidden_neurons, num_layers;
    int input_size, num_classes, num_input, batch_size;
    int nnz;
    float density;
    bool is_cifar;

    // SNICIT augmentation data structures
    int threshold;
    int *y_star_row;
    int seed_size;
    int *centroid_LUT;
    bool *ne_record;
    int *rowsY;

    void _infer();
    
    void _preprocess(const std::string& input_path);
        
    void _weight_bias_alloc_read();

    void _input_alloc_read(const std::string& input_path);

    void _result_alloc_read(const std::string& input_path);

  public:
    SNICIT(
      const std::string& _weight_path,
      const std::string& _bias_path,
      const int _num_hidden_neurons,
      const int _num_layers,
      const float _density,
      const int _seed_size,
      const int _threshold,
      const int _batch_size,
      const int _num_input,
      const bool _is_cifar
    );

    ~SNICIT();
    
    void infer(const std::string& input_path);

};

SNICIT::SNICIT(
    const std::string& _weight_path,
    const std::string& _bias_path,
    const int _num_hidden_neurons,
    const int _num_layers,
    const float _density,
    const int _seed_size,
    const int _threshold,
    const int _batch_size,
    const int _num_input,
    const bool _is_cifar
) : weight_path(_weight_path), bias_path(_bias_path), 
    num_hidden_neurons(_num_hidden_neurons), num_layers(_num_layers), 
    num_classes(10), density(_density), 
    nnz(std::round(_num_hidden_neurons*_num_hidden_neurons*_density)), num_input(_num_input), batch_size(_batch_size),
    seed_size(_seed_size), threshold(_threshold), is_cifar(_is_cifar)
 {
  std::cout<<"Constructing SNICIT method......\n";
  input_size = is_cifar ? _num_hidden_neurons : 784;
}

SNICIT::~SNICIT() {
  for(auto& each_Y : _dev_Y_hidden) {
    checkHip(hipFree(each_Y));
  }
  for(auto& each_dev_hidden_roffw : _dev_hidden_roffw) {
    checkHip(hipFree(each_dev_hidden_roffw));
  }
  for(auto& each_dev_hidden_colsw : _dev_hidden_colsw) {
    checkHip(hipFree(each_dev_hidden_colsw));
  }
  for(auto& each_dev_hidden_valsw : _dev_hidden_valsw) {
    checkHip(hipFree(each_dev_hidden_valsw));
  }
  for(auto& each_dev_hidden_bias : _dev_hidden_bias) {
    checkHip(hipFree(each_dev_hidden_bias));
  }
  if (!is_cifar) {
    checkHip(hipFree(_dev_Y_input));
    checkHip(hipFree(_dev_input_weight));
    checkHip(hipFree(_dev_input_bias));
  }
  checkHip(hipFree(_dev_output_weight));
  checkHip(hipFree(_dev_output_bias));
  
  checkHip(hipFree(_dev_Y_output));
  checkHip(hipFree(_dev_Y_output_whole));
  checkHip(hipFree(_dev_result_label));
  checkHip(hipFree(y_star_row));
  checkHip(hipFree(centroid_LUT));
  checkHip(hipFree(ne_record));
  checkHip(hipFree(rowsY));

  delete [] Y_input;
}

void SNICIT::infer(const std::string& input_path) {
  _preprocess(input_path);

  _infer();

}

void SNICIT::_preprocess(const std::string& input_path) {
  std::cout<<"preprocessing......\n";
  auto _tic = std::chrono::steady_clock::now();

  _weight_bias_alloc_read();

  _input_alloc_read(input_path);

  _result_alloc_read(input_path);

  checkHip(hipMallocManaged(
    &y_star_row,
    seed_size * sizeof(int)
  ));

  checkHip(hipMallocManaged(
    &centroid_LUT,
    batch_size * sizeof(int)
  ));
  checkHip(hipMallocManaged(
    &ne_record,
    batch_size * sizeof(bool)
  ));
  checkHip(hipMallocManaged(
    &rowsY,
    batch_size * sizeof(int)
  ));

  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic).count();
  std::cout<<"finished preprocessing in "<<duration<< "ms"<<std::endl;
}


void SNICIT::_weight_bias_alloc_read() {
  std::string line;
  std::ifstream MyReadFile;
  int ptr = 0;
  int file_offset;
  if (!is_cifar) {
    file_offset = 2;
    // allocate input layer's weight and bias
    checkHip(hipMalloc(
      &_dev_input_weight,
      input_size * num_hidden_neurons * sizeof(float)
    ));
    checkHip(hipMalloc(
      &_dev_input_bias,
      num_hidden_neurons * sizeof(float)
    ));
    // read input layer's weight and bias
    float *input_weight;
    float *input_bias;
    input_weight = new float[input_size * num_hidden_neurons];
    input_bias = new float[num_hidden_neurons];

    MyReadFile = std::ifstream(weight_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      while(std::getline(MyReadFile, line)){
        input_weight[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    MyReadFile = std::ifstream(bias_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        input_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy input layer's weight and bias
    checkHip(hipMemcpy(_dev_input_weight, input_weight, 
      input_size * num_hidden_neurons * sizeof(float), hipMemcpyHostToDevice));
    checkHip(hipMemcpy(_dev_input_bias, input_bias, 
      num_hidden_neurons * sizeof(float), hipMemcpyHostToDevice));
    delete [] input_weight;
    delete [] input_bias;
  }
  else {file_offset = 1;}
  for(int hidden_layer = 0; hidden_layer < num_layers; hidden_layer++) {
    int *hidden_roffw;
    int *hidden_colsw;
    float *hidden_valsw;
    float *hidden_bias;

    hidden_roffw = new int[num_hidden_neurons+1];
    hidden_colsw = new int[nnz];
    hidden_valsw = new float[nnz];
    hidden_bias = new float[num_hidden_neurons];
    memset(hidden_roffw, 0, (num_hidden_neurons+1)*sizeof(int));
    int* dev_cur_roffw;
    int* dev_cur_colsw;
    float* dev_cur_valsw;
    float* dev_cur_bias;

    // allocate hidden layer's weight and bias
    checkHip(hipMallocManaged(
      &dev_cur_roffw,
      (num_hidden_neurons+1) * sizeof(float)
    ));
    checkHip(hipMallocManaged(
      &dev_cur_colsw,
      nnz * sizeof(float)
    ));
    checkHip(hipMallocManaged(
      &dev_cur_valsw,
      nnz * sizeof(float)
    ));

    // read hidden layer
    
    MyReadFile = std::ifstream(weight_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;

      std::vector<std::string> tokens;
      while(std::getline(MyReadFile, line)){

        std::stringstream lineStream(line);
        std::string token;
        tokens.clear();
        while(std::getline(lineStream, token, '\t')) {
          tokens.push_back(std::move(token));
        }

        hidden_roffw[std::stoi(tokens[0])+1]++;
        hidden_colsw[ptr] = std::stoi(tokens[1]);
        hidden_valsw[ptr] = std::stof(tokens[2]);
        ptr++;
      }
      for (int i = 0; i < num_hidden_neurons; i++)
      {
          hidden_roffw[i + 1] += hidden_roffw[i];
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy hidden layer
    checkHip(hipMemcpy(dev_cur_roffw, hidden_roffw, 
      (num_hidden_neurons+1) * sizeof(int), hipMemcpyHostToDevice));
    checkHip(hipMemcpy(dev_cur_colsw, hidden_colsw, 
      nnz * sizeof(int), hipMemcpyHostToDevice));
    checkHip(hipMemcpy(dev_cur_valsw, hidden_valsw, 
      nnz * sizeof(float), hipMemcpyHostToDevice));
    _dev_hidden_roffw.emplace_back(dev_cur_roffw);
    _dev_hidden_colsw.emplace_back(dev_cur_colsw);
    _dev_hidden_valsw.emplace_back(dev_cur_valsw);

    checkHip(hipMallocManaged(
      &dev_cur_bias,
      num_hidden_neurons * sizeof(float)
    ));


    MyReadFile = std::ifstream(bias_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        hidden_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << bias_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();

    checkHip(hipMemcpy(dev_cur_bias, hidden_bias, 
      num_hidden_neurons * sizeof(float), hipMemcpyHostToDevice));

    _dev_hidden_bias.emplace_back(dev_cur_bias);

    delete [] hidden_roffw;
    delete [] hidden_colsw;
    delete [] hidden_valsw;
    delete [] hidden_bias;
  }

  checkHip(hipMalloc(
    &_dev_output_weight,
    num_hidden_neurons * num_classes * sizeof(float)
  ));
  checkHip(hipMalloc(
    &_dev_output_bias,
    num_classes * sizeof(float)
  ));
  float *output_weight;
  float *output_bias;
  output_weight = new float[num_hidden_neurons * num_classes];
  output_bias = new float[num_classes];
  MyReadFile = std::ifstream(weight_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_weight[ptr++] = std::stof(line);
  }
  MyReadFile.close();
  MyReadFile = std::ifstream(bias_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  ptr = 0;
  while(std::getline(MyReadFile, line)){
    output_bias[ptr++] = std::stof(line);
  }
  MyReadFile.close();

  checkHip(hipMemcpy(_dev_output_weight, output_weight, 
    num_hidden_neurons * num_classes * sizeof(float), hipMemcpyHostToDevice));
  checkHip(hipMemcpy(_dev_output_bias, output_bias, 
    num_classes * sizeof(float), hipMemcpyHostToDevice));

  delete [] output_weight;
  delete [] output_bias;
}

void SNICIT::_input_alloc_read(const std::string& input_path) {
  
  Y_input = new float[num_input*input_size];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      file.read((char*)&n_rows,sizeof(n_rows));
      file.read((char*)&n_cols,sizeof(n_cols));
      for(int i=0;i<10000;++i) // !!!!!!!!!!!!!!!!!! change!!!!!!!!!!!!!
      {
        for(int r=0;r<input_size;++r)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          Y_input[i*input_size+r]= ((float)temp)/255.0;
        }
      }
      checkHip(hipMallocManaged(
        &_dev_Y_input,
        batch_size*input_size * sizeof(float)
      ));
    }
    else {
      std::cout << "ERROR: MNIST input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file(input_path+"cifar-input.txt");
    if (file.is_open()) {
      std::string line;
      int ptr = 0;
      while(std::getline(file, line)){
        Y_input[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: CIFAR-10 input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }

  for(int buff=0; buff<2; buff++) {
    float *_dev_buff_Y;
    checkHip(hipMallocManaged(
      &_dev_buff_Y,
      batch_size*num_hidden_neurons * sizeof(float)
    ));
    _dev_Y_hidden.emplace_back(_dev_buff_Y);
  }
  checkHip(hipMallocManaged(
    &_dev_Y_output_whole,
    num_input*num_classes * sizeof(float)
  ));

  checkHip(hipMallocManaged(
    &_dev_Y_output,
    batch_size*num_classes * sizeof(float)
  ));
}

void SNICIT::_result_alloc_read(const std::string& input_path) {
  int *label;
  label = new int[num_input];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-labels-idx1-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      for(int i = 0; i < 10000; ++i)
      {
        unsigned char temp=0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
      }
    }
    else {
      std::cout << "ERROR: MNIST result file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file;
    file.open(input_path+"cifar-label.bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "ERROR: CIFAR-10 result file open failed" << std::endl;
        exit(1);
    }

    auto file_size = file.tellg();
    auto buffer = std::make_unique<char[]>(file_size);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    for(int i = 0; i < num_input; ++i){
      int l = buffer[i * 3073];
      label[i] = l;
    }
  }

  checkHip(hipMallocManaged(
    &_dev_result_label,
    num_input * sizeof(int)
  ));

  checkHip(hipMemcpy(_dev_result_label, label, 
    num_input * sizeof(int), hipMemcpyHostToDevice));

  delete [] label;
}

void SNICIT::_infer() {
  std::cout<<"inferring......\n";
  auto _tic = std::chrono::steady_clock::now();
  double sparse_duration = 0.0;
  hipStream_t dev_stream;
  double post_duration = 0.0;
  checkHip(hipStreamCreate(&dev_stream));
  for (int round = 0; round < num_input / batch_size; round++) {
    std::cout<<"[round "<<round<<"] begins: "<<std::endl;
    if (!is_cifar) {
      checkHip(hipMemcpy(_dev_Y_input, Y_input+round * batch_size * input_size, 
        batch_size*input_size * sizeof(float), hipMemcpyHostToDevice));
      dense_input<<<batch_size, dim3(num_hidden_neurons, (int)(1024/num_hidden_neurons), 1), 
        sizeof(float)*num_hidden_neurons, dev_stream>>>(_dev_Y_input, 
        _dev_input_weight, _dev_input_bias, batch_size, input_size, num_hidden_neurons, _dev_Y_hidden[0]);
    }
    else {
      checkHip(hipMemcpy(_dev_Y_hidden[0], Y_input+round * batch_size * input_size, 
        batch_size*input_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    
    checkHip(hipStreamSynchronize(dev_stream));
    auto sparse_tic = std::chrono::steady_clock::now();
    // pre-convergence
    auto pre_tic = std::chrono::steady_clock::now();
    for(int cur_layer = 0; cur_layer < threshold; cur_layer++) { // num_layers-2
      sparse_hidden<<<batch_size, dim3((int)(1024/num_hidden_neurons), num_hidden_neurons, 1), 
        sizeof(float)*num_hidden_neurons, dev_stream>>>(_dev_Y_hidden[cur_layer%2], 
        _dev_hidden_roffw[cur_layer], _dev_hidden_colsw[cur_layer], _dev_hidden_valsw[cur_layer], 
        _dev_hidden_bias[cur_layer], batch_size, num_hidden_neurons, num_hidden_neurons, _dev_Y_hidden[(cur_layer+1)%2]);
      checkHip(hipStreamSynchronize(dev_stream));

      checkHip(hipMemset(
        _dev_Y_hidden[cur_layer % 2],
        0,
        batch_size*num_hidden_neurons*sizeof(float)
      ));
    }
    auto pre_toc = std::chrono::steady_clock::now();
    auto pre_duration = std::chrono::duration_cast<std::chrono::microseconds>(pre_toc - pre_tic).count();
    std::cout<<"[**pre conv**] "<< pre_duration/1000.0<< "ms"<<std::endl;
    // y star generation
    auto cluster_tic = std::chrono::steady_clock::now();

    y_star_gen<<<1, dim3(1024/seed_size, seed_size, 1), 
      sizeof(float)*(num_hidden_neurons+2*seed_size), dev_stream>>>(_dev_Y_hidden[threshold % 2], y_star_row, batch_size, 
      num_hidden_neurons, seed_size);
    checkHip(hipStreamSynchronize(dev_stream));
    //checkHip(hipDeviceSynchronize());

    int y_star_cnt = 0;
    for (int i = 0; i < seed_size; i++) {
      if (y_star_row[i] != -1) {
        centroid_LUT[y_star_row[i]] = -1;
        y_star_row[y_star_cnt++] = y_star_row[i];
        // printf("%drow, label=%d\n", y_star_row[i], _dev_result_label[y_star_row[i]]);
      }
    }
    // coarse cluster
    coarse_cluster<<<batch_size, dim3(1024/y_star_cnt, y_star_cnt, 1), sizeof(float)*(num_hidden_neurons+y_star_cnt), dev_stream>>>
    (_dev_Y_hidden[threshold % 2], y_star_row, ne_record, y_star_cnt, centroid_LUT, num_hidden_neurons);

    checkHip(hipStreamSynchronize(dev_stream));
    //checkHip(hipDeviceSynchronize());
    int ne_rows = 0;
    for (int i = 0; i < batch_size; i++) {
      if (ne_record[i] != false) {
        rowsY[ne_rows++] = i;
      }
    }
    // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
    auto cluster_toc = std::chrono::steady_clock::now();
    auto cluster_duration = std::chrono::duration_cast<std::chrono::microseconds>(cluster_toc - cluster_tic).count();
    std::cout<<"[**cluster-based conversion**] finished in "<< cluster_duration/1000.0<< "ms"<<std::endl;
    // post convergence
    auto post_tic = std::chrono::steady_clock::now();
    for(int cur_layer = threshold; cur_layer < num_layers; cur_layer++) { // num_layers-2
      auto post_p_tic = std::chrono::steady_clock::now();
      sparse_hidden_post<<<ne_rows, dim3((int)(1024/num_hidden_neurons), num_hidden_neurons, 1), 
        sizeof(float)*num_hidden_neurons, dev_stream>>>(rowsY, _dev_Y_hidden[threshold % 2], 
        _dev_hidden_roffw[cur_layer], _dev_hidden_colsw[cur_layer], _dev_hidden_valsw[cur_layer], 
        batch_size, num_hidden_neurons, num_hidden_neurons, _dev_Y_hidden[(1+threshold) % 2]);
      checkHip(hipStreamSynchronize(dev_stream));
      update_post<<<ne_rows, num_hidden_neurons, 0,  dev_stream>>>(
        rowsY, centroid_LUT, _dev_Y_hidden[(1+threshold) % 2], _dev_hidden_bias[cur_layer], 
        num_hidden_neurons, ne_record, _dev_Y_hidden[threshold % 2]
      );
      checkHip(hipStreamSynchronize(dev_stream));
      //checkHip(hipDeviceSynchronize());
      int new_ne_rows = 0;
      for (int i = 0; i < ne_rows; i++) {
        if (ne_record[rowsY[i]] != false) {
          rowsY[new_ne_rows++] = rowsY[i];
        }
      }
      ne_rows = new_ne_rows;
      // std::cout<<"non empty rows in delta Y =  "<< ne_rows <<std::endl;
      checkHip(hipMemset(
        _dev_Y_hidden[(1+threshold) % 2],
        0,
        batch_size*num_hidden_neurons*sizeof(float)
      ));

      auto post_p_toc = std::chrono::steady_clock::now();
      auto post_p_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_p_toc - post_p_tic).count();
      // std::cout<<"[**post convergence**]finished layer "<< cur_layer <<" in "<< post_p_duration/1000.0<< "ms"<<std::endl;
    }
    auto post_toc = std::chrono::steady_clock::now();
    post_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_toc - post_tic).count();
    std::cout<<"[**post convergence**]finished in "<< post_duration/1000.0<< "ms"<<std::endl;
    // recovery
    auto recovery_tic = std::chrono::steady_clock::now();
    recover<<<batch_size, num_hidden_neurons, sizeof(float)*num_hidden_neurons, dev_stream>>>(_dev_Y_hidden[threshold % 2], centroid_LUT, num_hidden_neurons);
    checkHip(hipStreamSynchronize(dev_stream));
    auto recovery_toc = std::chrono::steady_clock::now();
    auto recovery_duration = std::chrono::duration_cast<std::chrono::microseconds>(recovery_toc - recovery_tic).count();
    std::cout<<"[**recovery**]finished in "<< recovery_duration/1000.0<< "ms"<<std::endl;
    auto sparse_toc = std::chrono::steady_clock::now();
    sparse_duration += std::chrono::duration_cast<std::chrono::microseconds>(sparse_toc - sparse_tic).count();

    // output layer
    dense_output<<<batch_size, dim3(num_classes, (int)(1024/num_classes), 1), 
      sizeof(float)*num_classes, dev_stream>>>(_dev_Y_hidden[threshold % 2], 
      _dev_output_weight, _dev_output_bias, batch_size, num_hidden_neurons, num_classes, _dev_Y_output);
    checkHip(hipStreamSynchronize(dev_stream));
    checkHip(hipMemcpy(_dev_Y_output_whole+round * batch_size * num_classes, _dev_Y_output,
      batch_size*num_classes * sizeof(float), hipMemcpyDeviceToDevice));
  }
  // std::cout<<"SNICIT runtime: "<< sparse_duration/1000.0<< "ms"<<std::endl;
  int* cnt;
  checkHip(hipMallocManaged(
    &cnt,
    sizeof(int)
  ));
  check_acc<<<1, 1024, sizeof(int), dev_stream>>>(_dev_Y_output_whole, num_classes, num_input, _dev_result_label, cnt);
  checkHip(hipStreamSynchronize(dev_stream));
  //checkHip(hipDeviceSynchronize());
  
  std::cout<<"SNICIT info: accuracy "<<100*((float)cnt[0]/(float)num_input)<<"%"<<" runtime "<< sparse_duration/1000.0<< "ms"
  <<" avgpost "<<post_duration/(1000*(num_layers-threshold))<< "ms"<<std::endl;
  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"[Total] finished inferring in "<<duration/1000.0<< "ms"<<std::endl;
  
}

}
