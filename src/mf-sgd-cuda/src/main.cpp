#include "sgd.h"

using namespace std;

bool is_numerical(char *str)
{
  int c = 0;
  while(*str != '\0')
  {
    if(isdigit(*str))
      c++;
    str++;
  }
  return c > 0;
}

Argument parse_argument(int argc, char **argv)
{
  vector<string> args;
  for(int i = 0; i < argc; i++)
    args.push_back(string(argv[i]));

  if(argc == 1)
    throw invalid_argument("error");

  Argument arg;
  int i;
  for(i = 1;i < argc; i++)
  {
    if(args[i].compare("-g") == 0)
    {
      if((i + 1) >= argc)
        throw invalid_argument("need to specify the id of GPUs\
            after -g");
      i++;
      if(!is_numerical(argv[i]))
        throw invalid_argument("-g should be followed by a number");
      arg.param.gpu = atoi(argv[i]);

    }
    else if(args[i].compare("-l") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify lambda after -l");
      i++;

      char *pch = strtok(argv[i], ",");
      if(!is_numerical(pch))
        throw invalid_argument("regularization coefficient should be a number");
      arg.param.lambda_p = (SGDRate)strtod(pch, NULL);
      arg.param.lambda_q = (SGDRate)strtod(pch, NULL);
      pch = strtok(NULL, ",");
      if(pch != NULL)
      {
        if(!is_numerical(pch))
          throw invalid_argument("regularization coefficient should be a number");
        arg.param.lambda_q = (SGDRate)strtod(pch, NULL);
      }
    }
    else if(args[i].compare("-k") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of factors after -k");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-k should be followed by a number");
      arg.param.k = atoi(argv[i]);
    }
    else if(args[i].compare("-t") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of iterations after -t");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-i should be followed by a number");
      arg.param.num_iters = atoi(argv[i]);
    }
    else if(args[i].compare("-r") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify eta after -r");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-r should be followed by a number");
      arg.param.lrate = (SGDRate)atof(argv[i]);
    }
    else if(args[i].compare("-a") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify eta after -a");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-a should be followed by a number");
      arg.param.alpha = (SGDRate)atof(argv[i]);
    }
    else if(args[i].compare("-b") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify eta after -b");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-b should be followed by a number");
      arg.param.beta = (SGDRate)atof(argv[i]);
    }
    else if(args[i].compare("-s") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of parallel workers\
            after -s(multiples of 4)");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-s should be followed by a number which is multiple of 4");
      arg.param.num_workers = ((atoi(argv[i]) + 3)/4)*4;
    }
    else if(args[i].compare("-u") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of u grid after -u");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-u should be followed by a number");
      arg.param.u_grid = atoi(argv[i]);
    }
    else if(args[i].compare("-v") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of v grid after -v");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-v should be followed by a number");
      arg.param.v_grid = atoi(argv[i]);
    }
    else if(args[i].compare("-x") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of x grid\
            after -x");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-x should be followed by a number");
      arg.param.x_grid = atoi(argv[i]);
    }
    else if(args[i].compare("-y") == 0)
    {
      if((i+1) >= argc)
        throw invalid_argument("need to specify number of y grid after -y");
      i++;

      if(!is_numerical(argv[i]))
        throw invalid_argument("-y should be followed by a number");
      arg.param.y_grid = atoi(argv[i]);
    }
    else if(args[i].compare("-p") == 0)
    {
      if(i == argc-1)
        throw invalid_argument("need to specify path after -p");
      i++;

      arg.va_path = string(args[i]);
    }
    else break;
  }

  if(i >= argc)
    throw invalid_argument("training data not specified");

  arg.tr_path = string(args[i++]);

  if(i < argc)
  {
    arg.model_path = string(args[i]);
  }
  else if(i == argc)
  {
    const char *ptr = strrchr(&*arg.tr_path.begin(), '/');
    if(!ptr)
      ptr = arg.tr_path.c_str();
    else
      ++ptr;
    arg.model_path = string(ptr) + ".model";
  }
  else
  {
    throw invalid_argument("invalid argument");
  }

  // these must be positive: they are used as divisors and array sizes
  // (e.g. grid segmentation and gridSize allocation), so a value <= 0
  // would cause a division-by-zero or zero-size allocation later on.
  if(arg.param.k <= 0)
    throw invalid_argument("number of factors (-k) must be positive");
  if(arg.param.num_iters <= 0)
    throw invalid_argument("number of iterations (-t) must be positive");
  if(arg.param.num_workers <= 0)
    throw invalid_argument("number of workers (-s) must be positive");
  if(arg.param.u_grid <= 0 || arg.param.v_grid <= 0 ||
     arg.param.x_grid <= 0 || arg.param.y_grid <= 0)
    throw invalid_argument("grid sizes (-u/-v/-x/-y) must be positive");

  if(arg.param.u_grid*arg.param.v_grid == 1)
  {
    arg.param.x_grid = 1;
    arg.param.y_grid = 1;
  }
  arg.param.ux = arg.param.u_grid*arg.param.x_grid;
  arg.param.vy = arg.param.v_grid*arg.param.y_grid;

  return arg;
}

//save the p & q model into file.
int save_model(mf_model const *model, char const *path)
{
  printf("save_model ...\n");

  char command[1024];
  sprintf(command, "rm -f %s", path);
  int sys_ret = system(command);

  FILE* fptr = fopen(path, "w");
  if(fptr == NULL)
  {
    printf("save model failed\n");
    return 1;
  }

  int f = 0;
  fwrite(&f, sizeof(int), 1, fptr);
  fwrite(&(model->m), sizeof(int), 1, fptr);
  fwrite(&(model->n), sizeof(int), 1, fptr);
  fwrite(&(model->k), sizeof(int), 1, fptr);
  fwrite(&(model->b), sizeof(float), 1, fptr);

  auto write = [&] (float *ptr, int size)
  {
    for(SGDIndex i = 0; i < size; i++)
    {
      SGDRate *ptr1 = ptr + (long long)i*model->k;
      size_t write_size = fwrite(ptr1, sizeof(float), model->k, fptr);
    }
  };
  printf("saving feature p(%d)...\n", model->m);
  write(model->floatp, model->m);
  printf("saving feature q(%d)...\n", model->n);
  write(model->floatq, model->n);

  fclose(fptr);
  return 0;
}

int main(int argc, char**argv)
{

  Argument arg;
  try
  {  
    arg = parse_argument(argc,argv);
    arg.print_arg();

    fflush(stdout);
  }
  catch(invalid_argument &e)
  {
    cout << e.what() << endl;
    return 1;
  }

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if(error_id != cudaSuccess || deviceCount <= 0)
  {
    fprintf(stderr, "no CUDA capable device found: %s\n",
        cudaGetErrorString(error_id));
    return 1;
  }
  cudaSetDevice(arg.param.gpu%deviceCount);

  mf_problem tr,va;

  tr = read_problem(arg.tr_path);
  tr.u_grid = arg.param.u_grid;
  tr.v_grid = arg.param.v_grid;
  tr.x_grid = arg.param.x_grid;
  tr.y_grid = arg.param.y_grid;
  tr.ux = arg.param.ux;
  tr.vy = arg.param.vy;

  mf_model* model = sgd_train(&tr, &va, arg.param);

  save_model(model, arg.model_path.c_str());

  cudaFreeHost(model->floatp);
  cudaFreeHost(model->floatq);
  cudaFreeHost(model->halfp);
  cudaFreeHost(model->halfq);
  cudaFreeHost(tr.R);

  printf("\ntraining application finished...\n\n\n");

  return 0;
}
