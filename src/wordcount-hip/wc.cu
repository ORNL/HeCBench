#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <vector>
#include <numeric>

// This example computes the number of words in a text sample
// with a single call to thrust::inner_product.  The algorithm
// counts the number of characters which start a new word, i.e.
// the number of characters where input[i] is an alphabetical
// character and input[i-1] is not an alphabetical character.


// determines whether the character is alphabetical
__host__ __device__
bool is_alpha(const char c)
{
  return (c >= 'A' && c <= 'z');
}

// determines whether the right character begins a new word
struct is_word_start
{
  __host__ __device__
  bool operator()(const char& left, const char& right) const
  {
    return is_alpha(right) && !is_alpha(left);
  }
};


int word_count(const std::vector<char> &input)
{
  // check for empty string
  if (input.empty()) return 0;

  // transfer to device
  thrust::device_vector<char> d_input(input.begin(), input.end());

  // compute the number characters that start a new word
  int wc = thrust::inner_product(d_input.begin(), d_input.end() - 1,  // sequence of left characters
      d_input.begin() + 1,             // sequence of right characters
      0,                               // initialize sum to 0
      thrust::plus<int>(),             // sum values together
      is_word_start());                // how to compare the left and right characters

  // if the first character is alphabetical, then it also begins a word
  if (is_alpha(d_input.front())) wc++;

  return wc;
}

int word_count_reference(const std::vector<char> &input)
{
  // check for empty string
  if (input.empty()) return 0;

  // compute the number characters that start a new word
  int wc = std::inner_product(
      input.cbegin(), input.cend() - 1, // sequence of left characters
      input.cbegin() + 1,               // sequence of right characters
      0,                                // initialize sum to 0
      std::plus<int>(),                 // sum values together
      is_word_start());                 // how to compare the left and right characters

  // if the first character is alphabetical, then it also begins a word
  if (is_alpha(input.front())) wc++;

  return wc;
}
